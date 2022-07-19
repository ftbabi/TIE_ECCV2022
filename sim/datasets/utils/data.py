import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from .misc import rand_float, rand_int, find_relations_neighbor, normalize

from mmcv.parallel import DataContainer


def sample_control_RiceGrip():
    dis = np.random.rand() * 0.5
    angle = np.random.rand() * np.pi * 2.
    x = np.cos(angle) * dis
    z = np.sin(angle) * dis
    d = np.random.rand() * 0.3 + 0.7    # (0.6, 0.9)
    return x, z, d

def sample_control_FluidShake(x_box, time_step, dt):
    control = np.zeros(time_step)
    v_box = 0.
    for step in range(time_step):
        control[step] = v_box
        x_box += v_box * dt
        v_box += rand_float(-0.15, 0.15) - x_box * 0.1
    return control

def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_var(axis, angle):
    axis /= torch.norm(axis)

    half = angle * 0.5
    w = torch.cos(half)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two

    quat = torch.cat([axis, w])
    # print("quat size", quat.size())

    return quat

def calc_shape_states_RiceGrip(t, dt, shape_state_dim, gripper_config):
    rest_gripper_dis = 1.8
    x, z, d = gripper_config
    s = (rest_gripper_dis - d) / 2.
    half_rest_gripper_dis = rest_gripper_dis / 2.

    time = max(0., t) * 5
    lastTime = max(0., t - dt) * 5

    states = np.zeros((2, shape_state_dim))

    dis = np.sqrt(x**2 + z**2)
    angle = np.array([-z / dis, x / dis])
    quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(x / z))

    e_0 = np.array([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
    e_1 = np.array([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

    e_0_curr = e_0 + angle * np.sin(time) * s
    e_1_curr = e_1 - angle * np.sin(time) * s
    e_0_last = e_0 + angle * np.sin(lastTime) * s
    e_1_last = e_1 - angle * np.sin(lastTime) * s

    states[0, :3] = np.array([e_0_curr[0], 0.6, e_0_curr[1]])
    states[0, 3:6] = np.array([e_0_last[0], 0.6, e_0_last[1]])
    states[0, 6:10] = quat
    states[0, 10:14] = quat

    states[1, :3] = np.array([e_1_curr[0], 0.6, e_1_curr[1]])
    states[1, 3:6] = np.array([e_1_last[0], 0.6, e_1_last[1]])
    states[1, 6:10] = quat
    states[1, 10:14] = quat

    return states


def calc_shape_states_RiceGrip_var(t, dt, gripper_config):
    rest_gripper_dis = Variable(torch.FloatTensor([1.8]).cuda())
    x, z, d = gripper_config[0:1], gripper_config[1:2], gripper_config[2:3]

    s = (rest_gripper_dis - d) / 2.
    half_rest_gripper_dis = rest_gripper_dis / 2.

    time = max(0., t) * 5
    lastTime = max(0., t - dt) * 5

    dis = torch.sqrt(x**2 + z**2)
    angle = torch.cat([-z / dis, x / dis])
    quat = quatFromAxisAngle_var(Variable(torch.FloatTensor([0., 1., 0.]).cuda()), torch.atan(x / z))

    e_0 = torch.cat([x + z * half_rest_gripper_dis / dis, z - x * half_rest_gripper_dis / dis])
    e_1 = torch.cat([x - z * half_rest_gripper_dis / dis, z + x * half_rest_gripper_dis / dis])

    e_0_curr = e_0 + angle * np.sin(time) * s
    e_1_curr = e_1 - angle * np.sin(time) * s
    e_0_last = e_0 + angle * np.sin(lastTime) * s
    e_1_last = e_1 - angle * np.sin(lastTime) * s

    y = Variable(torch.FloatTensor([0.6]).cuda())
    states_0 = torch.cat([e_0_curr[0:1], y, e_0_curr[1:2], e_0_last[0:1], y, e_0_last[1:2], quat, quat])
    states_1 = torch.cat([e_1_curr[0:1], y, e_1_curr[1:2], e_1_last[0:1], y, e_1_last[1:2], quat, quat])

    # print(states_0.requires_grad, states_1.requires_grad)
    # print("gripper #0:", states_0.size())
    # print("gripper #1:", states_1.size())

    return torch.cat([states_0.view(1, -1), states_1.view(1, -1)], 0)


def calc_box_init_FluidShake(dis_x, dis_z, height, border):
    center = np.array([0., 0., 0.])
    quat = np.array([1., 0., 0., 0.])
    boxes = []

    # floor
    halfEdge = np.array([dis_x/2., border/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # left wall
    halfEdge = np.array([border/2., (height+border)/2., dis_z/2.])
    boxes.append([halfEdge, center, quat])

    # right wall
    boxes.append([halfEdge, center, quat])

    # back wall
    halfEdge = np.array([(dis_x+border*2)/2., (height+border)/2., border/2.])
    boxes.append([halfEdge, center, quat])

    # front wall
    boxes.append([halfEdge, center, quat])

    return boxes


def calc_shape_states_FluidShake(x_curr, x_last, box_dis, height, border):
    dis_x, dis_z = box_dis
    quat = np.array([1., 0., 0., 0.])

    states = np.zeros((5, 14))

    states[0, :3] = np.array([x_curr, border/2., 0.])
    states[0, 3:6] = np.array([x_last, border/2., 0.])

    states[1, :3] = np.array([x_curr-(dis_x+border)/2., (height+border)/2., 0.])
    states[1, 3:6] = np.array([x_last-(dis_x+border)/2., (height+border)/2., 0.])

    states[2, :3] = np.array([x_curr+(dis_x+border)/2., (height+border)/2., 0.])
    states[2, 3:6] = np.array([x_last+(dis_x+border)/2., (height+border)/2., 0.])

    states[3, :3] = np.array([x_curr, (height+border)/2., -(dis_z+border)/2.])
    states[3, 3:6] = np.array([x_last, (height+border)/2., -(dis_z+border)/2.])

    states[4, :3] = np.array([x_curr, (height+border)/2., (dis_z+border)/2.])
    states[4, 3:6] = np.array([x_last, (height+border)/2., (dis_z+border)/2.])

    states[:, 6:10] = quat
    states[:, 10:] = quat

    return states


def calc_shape_states_FluidShake_var(x_curr, x_last, box_dis, height, border):
    dis_x, dis_z = box_dis

    dis_x = Variable(torch.FloatTensor([dis_x]).cuda())
    dis_z = Variable(torch.FloatTensor([dis_z]).cuda())
    height = Variable(torch.FloatTensor([height]).cuda())
    border = Variable(torch.FloatTensor([border]).cuda())
    zero = Variable(torch.FloatTensor([0.]).cuda())
    quat = Variable(torch.FloatTensor([1., 0., 0., 0.]).cuda())

    state_0 = torch.cat([
        x_curr, border/2., zero, x_last, border/2., zero, quat, quat]).view(1, -1)

    state_1 = torch.cat([
        x_curr-(dis_x+border)/2., (height+border)/2., zero,
        x_last-(dis_x+border)/2., (height+border)/2., zero,
        quat, quat]).view(1, -1)

    state_2 = torch.cat([
        x_curr+(dis_x+border)/2., (height+border)/2., zero,
        x_last+(dis_x+border)/2., (height+border)/2., zero,
        quat, quat]).view(1, -1)

    state_3 = torch.cat([
        x_curr, (height+border)/2., -(dis_z+border)/2.,
        x_last, (height+border)/2., -(dis_z+border)/2.,
        quat, quat]).view(1, -1)

    state_4 = torch.cat([
        x_curr, (height+border)/2., (dis_z+border)/2.,
        x_last, (height+border)/2., (dis_z+border)/2.,
        quat, quat]).view(1, -1)

    states = torch.cat([state_0, state_1, state_2, state_3, state_4], 0)
    # print("states size", states.size())

    return states


def gen_PyFleX(info):

    env, root_num = info['env'], info['root_num']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, n_instance = info['n_rollout'], info['n_instance']
    time_step, time_step_clip = info['time_step'], info['time_step_clip']
    shape_state_dim, dt = info['shape_state_dim'], info['dt']
    scene_params = info['scene_params']

    env_idx = info['env_idx']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2**32)

    # positions, velocities
    if env_idx == 5:    # RiceGrip
        stats = [init_stat(6), init_stat(6)]
    else:
        stats = [init_stat(3), init_stat(3)]

    import pyflex
    pyflex.init()

    for i in range(n_rollout):

        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        if env == 'FluidFall':
            scene_params = np.zeros(1)
            pyflex.set_scene(env_idx, scene_params, thread_idx)
            n_particles = pyflex.get_n_particles()
            positions = np.zeros((time_step, n_particles, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles, 3), dtype=np.float32)

            # Get the first clip of particles' positions. Just to init to calculate velocity later
            for j in range(time_step_clip):
                # x, y, z, w
                p_clip = pyflex.get_positions().reshape(-1, 4)[:, :3]
                pyflex.step()

            for j in range(time_step):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                if j == 0:
                    velocities[j] = (positions[j] - p_clip) / dt
                else:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j]]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'BoxBath':
            # BoxBath

            assert scene_params is not None
            pyflex.set_scene(env_idx, scene_params, thread_idx)
            n_particles = pyflex.get_n_particles()
            positions = np.zeros((time_step, n_particles, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles, 3), dtype=np.float32)

            for j in range(time_step_clip):
                pyflex.step()

            rigid_idxs = pyflex.get_rigidIndices()
            rigid_particles_num = pyflex.get_n_rigidPositions()
            assert len(rigid_idxs) == rigid_particles_num
            # Get rigid position; Rigid is at [0, 64) index
            p = pyflex.get_positions().reshape(-1, 4)[:rigid_particles_num, :3]
            clusters = []
            st_time = time.time()
            # Cluster rigid particles
            kmeans = MiniBatchKMeans(n_clusters=root_num[0][0], random_state=0).fit(p)
            # print('Time on kmeans', time.time() - st_time)
            clusters.append([[kmeans.labels_]])
            # centers = kmeans.cluster_centers_

            ref_rigid = p

            for j in range(time_step):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                # Fix cubic rigid
                # # apply rigid projection to ground truth
                # XX = ref_rigid
                # YY = positions[j, :64]
                # # print("MSE init", np.mean(np.square(XX - YY)))

                # X = XX.copy().T
                # Y = YY.copy().T
                # mean_X = np.mean(X, 1, keepdims=True)
                # mean_Y = np.mean(Y, 1, keepdims=True)
                # X = X - mean_X
                # Y = Y - mean_Y
                # C = np.dot(X, Y.T)
                # U, S, Vt = np.linalg.svd(C)
                # D = np.eye(3)
                # D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                # R = np.dot(Vt.T, np.dot(D, U.T))
                # t = mean_Y - np.dot(R, mean_X)

                # YY_fitted = (np.dot(R, XX.T) + t).T
                # # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                # positions[j, :64] = YY_fitted

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], clusters]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'FluidShake':
            # if env is FluidShake
            # In case the water can get out of box
            assert scene_params is not None
            height = 2.0
            border = 0.025
            dim_x = rand_int(*scene_params[0])
            dim_y = rand_int(*scene_params[1])
            dim_z = scene_params[2]
            x_center = rand_float(-0.2, 0.2)
            x = x_center - (dim_x-1)/2.*0.055
            y = 0.055/2. + border + 0.01
            z = 0. - (dim_z-1)/2.*0.055
            box_dis_x = dim_x * 0.055 + rand_float(0., 0.3)
            box_dis_z = 0.2

            # For rigid params
            dim_x_r = scene_params[3]
            dim_y_r = scene_params[4]
            dim_z_r = scene_params[5]
            rigid_type = scene_params[6]
            rigid_invm = scene_params[7]

            x_r = x_center - dim_x_r / 2.
            y_r = y + dim_y * 0.052
            z_r = -dim_z_r / 2.
            final_scene_params = np.array([
                x, y, z, dim_x, dim_y, dim_z, # Fluid
                rigid_type, rigid_invm, x_r, y_r, z_r, dim_x_r, dim_y_r, dim_z_r,
                box_dis_x, box_dis_z])
            pyflex.set_scene(env_idx, final_scene_params, 0)

            # Create box with walls and floor to contain fluid and rigid
            # The quat for box is constant [1, 0, 0, 0]
            boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)

            for i in range(len(boxes)):
                halfEdge = boxes[i][0]
                center = boxes[i][1]
                quat = boxes[i][2]
                pyflex.add_box(halfEdge, center, quat)

            n_particles = pyflex.get_n_particles()
            n_shapes = pyflex.get_n_shapes()

            # print("n_particles", n_particles)
            # print("n_shapes", n_shapes)

            positions = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            velocities = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

            x_box = x_center
            v_box = 0.
            # Initialize, set the state of box
            for j in range(time_step_clip):
                x_box_last = x_box
                x_box += v_box * dt
                shape_states_ = calc_shape_states_FluidShake(
                    x_box, x_box_last, final_scene_params[-2:], height, border)
                pyflex.set_shape_states(shape_states_)
                pyflex.step()

            for j in range(time_step):
                # Let the box move
                x_box_last = x_box
                x_box += v_box * dt
                v_box += rand_float(-0.15, 0.15) - x_box * 0.1
                shape_states_ = calc_shape_states_FluidShake(
                    x_box, x_box_last, final_scene_params[-2:], height, border)
                pyflex.set_shape_states(shape_states_)

                # Get particles' states and the box's state
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, shape_state_dim)

                # Set box's state
                for k in range(n_shapes):
                    positions[j, n_particles + k] = shape_states[k, :3]
                    shape_quats[j, k] = shape_states[k, 6:10]

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], shape_quats[j], final_scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        elif env == 'RiceGrip':
            # if env is RiceGrip
            # repeat the grip for R times
            assert scene_params is not None

            R = 3
            gripper_config = sample_control_RiceGrip()

            if i % R == 0:
                ### set scene
                # x, y, z: [8.0, 10.0]
                # clusterStiffness: [0.3, 0.7]
                # clusterPlasticThreshold: [0.00001, 0.0005]
                # clusterPlasticCreep: [0.1, 0.3]
                x = rand_float(*scene_params[0])
                y = rand_float(*scene_params[1])
                z = rand_float(*scene_params[2])

                clusterStiffness = rand_float(0.4, 0.8)
                clusterPlasticThreshold = rand_float(0.00001, 0.0003)
                clusterPlasticCreep = rand_float(0.1, 0.3)

                final_scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
                pyflex.set_scene(env_idx, final_scene_params, thread_idx)
                final_scene_params[4] *= 1000.

                halfEdge = np.array([0.15, 0.8, 0.15])
                center = np.array([0., 0., 0.])
                quat = np.array([1., 0., 0., 0.])
                pyflex.add_box(halfEdge, center, quat)
                pyflex.add_box(halfEdge, center, quat)

                n_particles = pyflex.get_n_particles()
                n_shapes = pyflex.get_n_shapes()

                positions = np.zeros((time_step, n_particles + n_shapes, 6), dtype=np.float32)
                velocities = np.zeros((time_step, n_particles + n_shapes, 6), dtype=np.float32)
                shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

                for j in range(time_step_clip):
                    shape_states = calc_shape_states_RiceGrip(0, dt, shape_state_dim, gripper_config)
                    pyflex.set_shape_states(shape_states)
                    pyflex.step()

                p = pyflex.get_positions().reshape(-1, 4)[:, :3]

                clusters = []
                st_time = time.time()
                kmeans = MiniBatchKMeans(n_clusters=root_num[0][0], random_state=0).fit(p)
                # print('Time on kmeans', time.time() - st_time)
                clusters.append([[kmeans.labels_]])
                # centers = kmeans.cluster_centers_

            for j in range(time_step):
                shape_states = calc_shape_states_RiceGrip(j * dt, dt, shape_state_dim, gripper_config)
                pyflex.set_shape_states(shape_states)

                positions[j, :n_particles, :3] = pyflex.get_rigidGlobalPositions().reshape(-1, 3)
                positions[j, :n_particles, 3:] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                shape_states = pyflex.get_shape_states().reshape(-1, shape_state_dim)

                for k in range(n_shapes):
                    positions[j, n_particles + k, :3] = shape_states[k, :3]
                    positions[j, n_particles + k, 3:] = shape_states[k, :3]
                    shape_quats[j, k] = shape_states[k, 6:10]

                if j > 0:
                    velocities[j] = (positions[j] - positions[j - 1]) / dt

                pyflex.step()

                data = [positions[j], velocities[j], shape_quats[j], clusters, final_scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

        else:
            raise AssertionError("Unsupported env")

        # change dtype for more accurate stat calculation
        # only normalize positions and velocities
        datas = [positions.astype(np.float64), velocities.astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats

def prepare_input(data, stat, args, phases_dict, verbose=0, var=False, abs_point=False, fix_attr_dim=0):

    # Arrangement:
    # particles, shapes, roots

    n_abs = 0
    if args.env == 'RiceGrip':
        positions, velocities, shape_quats, clusters, scene_params = data
        if abs_point:
            positions = np.concatenate((positions, np.zeros((1, positions.shape[-1]))), axis=0)
            velocities = np.concatenate((velocities, np.zeros((1, velocities.shape[-1]))), axis=0)
            n_abs = 1
        n_shapes = shape_quats.size(0) if var else shape_quats.shape[0]
    elif args.env == 'FluidShake':
        positions, velocities, shape_quats, scene_params = data
        if abs_point:
            positions = np.concatenate((positions, np.zeros((1, positions.shape[-1]))), axis=0)
            velocities = np.concatenate((velocities, np.zeros((1, velocities.shape[-1]))), axis=0)
            n_abs = 1
        # The sate of walls: front, back, left, right, floor
        n_shapes = shape_quats.size(0) if var else shape_quats.shape[0]
        clusters = None
    elif args.env == 'BoxBath':
        positions, velocities, clusters = data
        n_shapes = 5
        if abs_point:
            positions = np.concatenate((positions, np.zeros((2, positions.shape[-1]))), axis=0)
            velocities = np.concatenate((velocities, np.zeros((2, velocities.shape[-1]))), axis=0)
            n_abs = 2
    elif args.env == 'FluidFall':
        positions, velocities = data
        n_shapes = 0
        if not args.baseline:
            n_shapes = 1
        if abs_point:
            n_abs = 1
            positions = np.concatenate((positions, np.zeros((n_abs, positions.shape[-1]))), axis=0)
            velocities = np.concatenate((velocities, np.zeros((n_abs, velocities.shape[-1]))), axis=0) 
        clusters = None

    # count_nodes: all number of nodes, including the env, such as the wall
    count_nodes = positions.size(0) if var else positions.shape[0]
    n_particles = count_nodes - n_shapes - n_abs

    if verbose:
        print("positions", positions.shape)
        print("velocities", velocities.shape)

        print("n_particles", n_particles)
        print("n_shapes", n_shapes)
        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            print("shape_quats", shape_quats.shape)

    ### instance idx
    #   instance_idx (n_instance + 1): start idx of instance
    if args.env == 'RiceGrip' or args.env == 'FluidShake':
        # Here, don't take boxbath into consideration, as the wall are added extrally.
        instance_idx = [0, n_particles]
    else:
        instance_idx = phases_dict["instance_idx"]
    if verbose:
        print("Instance_idx:", instance_idx)


    ### object attributes
    #   dim 10: [rigid, fluid, root_0, root_1, gripper_0, gripper_1, mass_inv,
    #            clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep]
    if fix_attr_dim > 0:
        if args.env == 'RiceGrip' or args.env == 'FluidFall':
            raise NotImplementedError
        attr = np.zeros((count_nodes, fix_attr_dim))
    else:
        attr = np.zeros((count_nodes, args.attr_dim))
    # no need to include mass for now
    # attr[:, 6] = positions[:, -1].data.cpu().numpy() if var else positions[:, -1] # mass_inv
    if args.env == 'RiceGrip':
        # clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep
        attr[:, -3:] = scene_params[-3:]


    ### construct relations
    Rr_idxs = []        # relation receiver idx list, N*2, Rr_idxs[x, 0] is the index of particles; Rr_idxs[i, 1] equals to i.
    Rs_idxs = []        # relation sender idx list
    Ras = []            # relation attributes list
    values = []         # relation value list (should be 1)
    node_r_idxs = []    # list of corresponding receiver node idx
    node_s_idxs = []    # list of corresponding sender node idx
    psteps = []         # propagation steps

    ##### add env specific graph components
    rels = []
    if args.env == 'RiceGrip':
        # nodes = np.arange(n_particles)
        for i in range(n_shapes):
            # object attr:
            # [fluid, root, gripper_0, gripper_1, abs_point, 
            #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
            attr[n_particles + i, 2 + i] = 1

            pos = positions.data.cpu().numpy() if var else positions
            dis = np.linalg.norm(
                pos[:n_particles, 3:6:2] - pos[n_particles + i, 3:6:2], axis=1)
            nodes = np.nonzero(dis < 0.3)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            gripper = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            rels += [np.stack([nodes, gripper, np.ones(nodes.shape[0])], axis=1)]
        
        for i in range(n_abs):
            attr[n_particles+n_shapes + i, 4 + i] = 1
            if phases_dict['material'][i] == 'fluid':
                abs_p = np.ones(n_particles, dtype=np.int) * (n_particles+n_shapes + i)
                nodes = np.arange(n_particles)
                rels += [np.stack([nodes, abs_p, np.ones(n_particles)], axis=1)]
                rels += [np.stack([abs_p, nodes, np.ones(n_particles)], axis=1)]
    elif args.env == 'FluidShake':
        # Add relations between walls and particles
        for i in range(n_shapes):
            # One hot. Each position for each wall
            # object attr:
            # [fluid, wall*5, abs_point]
            # Fixed: [rigid, fluid, wall*5, abs_point*2]
            if fix_attr_dim > 0:
                attr[n_particles + i, 2 + i] = 1
            else:
                attr[n_particles + i, 1 + i] = 1

            # Calculate distance between each wall and particles
            pos = positions.data.cpu().numpy() if var else positions
            if i == 0:
                # floor
                dis = pos[:n_particles, 1] - pos[n_particles + i, 1]
            elif i == 1:
                # left
                dis = pos[:n_particles, 0] - pos[n_particles + i, 0]
            elif i == 2:
                # right
                dis = pos[n_particles + i, 0] - pos[:n_particles, 0]
            elif i == 3:
                # back
                dis = pos[:n_particles, 2] - pos[n_particles + i, 2]
            elif i == 4:
                # front
                dis = pos[n_particles + i, 2] - pos[:n_particles, 2]
            else:
                raise AssertionError("more shapes than expected")
            # Select the particles idx which are close to the i-th wall
            nodes = np.nonzero(dis < 0.1)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            # Set relation with thresh
            wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            # add relations: [particle_reciever_idx, wall_sender_idx, relation_type]
            # Relation type for root-leaf are 0. leaf-leaf are 1.
            rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]

        for i in range(n_abs):
            if fix_attr_dim > 0:
                attr[n_particles+n_shapes + i, 8 + i] = 1 # the 7-th is for rigid abs. though this has no effect
            else:
                attr[n_particles+n_shapes + i, 6 + i] = 1
            st, ed = instance_idx[i], instance_idx[i + 1]
            # Rigid, then fluid;
            # The order is same with phase_dict['material']
            abs_p = np.ones(ed-st, dtype=np.int) * (n_particles+n_shapes + i)
            # The st == 0, thus no effect
            nodes = np.arange(ed-st) + st
            rels += [np.stack([nodes, abs_p, np.ones(ed-st)], axis=1)]
            rels += [np.stack([abs_p, nodes, np.ones(ed-st)], axis=1)]

    elif args.env == 'BoxBath':
        # Add relations between walls and particles
        for i in range(n_shapes):
            # One hot. Each position for each wall
            if args.hierarchy:
                attr[n_particles + i, 3 + i] = 1
            else:
                # rigid, fluid, wall*5
                attr[n_particles + i, 2 + i] = 1

            # Calculate distance between each wall and particles
            pos = positions.data.cpu().numpy() if var else positions
            if i == 0:
                # floor
                dis = pos[:n_particles, 1] - pos[n_particles + i, 1]
            elif i == 1:
                # left
                dis = pos[:n_particles, 0] - pos[n_particles + i, 0]
            elif i == 2:
                # right
                dis = pos[n_particles + i, 0] - pos[:n_particles, 0]
            elif i == 3:
                # back
                dis = pos[:n_particles, 2] - pos[n_particles + i, 2]
            elif i == 4:
                # front
                dis = pos[n_particles + i, 2] - pos[:n_particles, 2]
            else:
                raise AssertionError("more shapes than expected")
            # Select the particles idx which are close to the i-th wall
            nodes = np.nonzero(dis < 0.1)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            # Set relation with thresh
            wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            # add relations: [particle_reciever_idx, wall_sender_idx, relation_type]
            # Relation type for root-leaf are 0. leaf-leaf are 1.
            rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]

        for i in range(n_abs):
            assert not args.hierarchy
            attr[n_particles+n_shapes + i, 7 + i] = 1
            st, ed = instance_idx[i], instance_idx[i + 1]
            # Rigid, then fluid;
            # The order is same with phase_dict['material']
            abs_p = np.ones(ed-st, dtype=np.int) * (n_particles+n_shapes + i)
            nodes = np.arange(ed-st) + st
            rels += [np.stack([nodes, abs_p, np.ones(ed-st)], axis=1)]
            rels += [np.stack([abs_p, nodes, np.ones(ed-st)], axis=1)]
    elif args.env == 'FluidFall' and not args.baseline:
        # Add relations between walls and particles
        for i in range(n_shapes):
            # One hot. Each position for each wall
            # fluid, wall*1
            attr[n_particles + i, 1 + i] = 1

            # Calculate distance between each wall and particles
            pos = positions.data.cpu().numpy() if var else positions
            if i == 0:
                # floor
                dis = pos[:n_particles, 1] - pos[n_particles + i, 1]
            else:
                raise AssertionError("more shapes than expected")
            # Select the particles idx which are close to the i-th wall
            nodes = np.nonzero(dis < 0.1)[0]

            if verbose:
                visualize_neighbors(positions, positions, 0, nodes)
                print(np.sort(dis)[:10])

            # Set relation with thresh
            wall = np.ones(nodes.shape[0], dtype=np.int) * (n_particles + i)
            # add relations: [particle_reciever_idx, wall_sender_idx, relation_type]
            # Relation type for root-leaf are 0. leaf-leaf are 1.
            rels += [np.stack([nodes, wall, np.ones(nodes.shape[0])], axis=1)]

        for i in range(n_abs):
            attr[n_particles+n_shapes + i, 2 + i] = 1
            st, ed = instance_idx[i], instance_idx[i + 1]
            # Rigid, then fluid;
            # The order is same with phase_dict['material']
            abs_p = np.ones(ed-st, dtype=np.int) * (n_particles+n_shapes + i)
            nodes = np.arange(ed-st) + st
            rels += [np.stack([nodes, abs_p, np.ones(ed-st)], axis=1)]
            rels += [np.stack([abs_p, nodes, np.ones(ed-st)], axis=1)]

    if verbose and len(rels) > 0:
        print(np.concatenate(rels, 0).shape)

    ##### add relations between leaf particles
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]

        if verbose:
            print('instance #%d' % i, st, ed)

        # To find anchors that are close to the queries
        # queries are reciever, anchors are sender
        if args.env == 'BoxBath':
            if phases_dict['material'][i] == 'rigid':
                attr[st:ed, 0] = 1
                # Rigid idx
                queries = np.arange(st, ed)
                # Other idx instead of rigid
                # if not args.hierarchy: 
                #     anchors = np.arange(n_particles)
                # else:
                #     anchors = np.concatenate((np.arange(st), np.arange(ed, n_particles)))
                anchors = np.arange(n_particles)
            elif phases_dict['material'][i] == 'fluid':
                attr[st:ed, 1] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")
        elif args.env == 'FluidFall' or args.env == 'RiceGrip':
            if phases_dict['material'][i] == 'fluid':
                attr[st:ed, 0] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")
        elif args.env == 'FluidShake':
            if phases_dict['material'][i] == 'fluid':
                if fix_attr_dim > 0:
                    attr[st:ed, 1] = 1
                else:
                    attr[st:ed, 0] = 1
                queries = np.arange(st, ed)
                anchors = np.arange(n_particles)
            else:
                raise AssertionError("Unsupported materials")
        else:
            raise AssertionError("Unsupported materials")

        # st_time = time.time()
        pos = positions
        pos = pos[:, -3:]
        # Reture the relations in format of: [particle_reciever_idx, wall_sender_idx, relation_type]
        rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
        # print("Time on neighbor search", time.time() - st_time)

    if verbose:
        print("Attr shape (after add env specific graph components):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))

    rels = np.concatenate(rels, 0)
    if rels.shape[0] > 0:
        if verbose:
            print("Relations neighbor", rels.shape)
        Rr_idxs.append(torch.LongTensor([rels[:, 0], np.arange(rels.shape[0])]))
        Rs_idxs.append(torch.LongTensor([rels[:, 1], np.arange(rels.shape[0])]))
        # For the attr of the leaves' relations, the attr are 0.
        Ra = np.zeros((rels.shape[0], args.relation_dim))
        Ras.append(torch.FloatTensor(Ra))
        values.append(torch.FloatTensor([1] * rels.shape[0]))
        
        if abs_point:
            r_idx = np.arange(n_particles+n_abs)
            for i in range(n_abs):
                r_idx[n_particles+i] = n_particles+n_shapes+i
            # r_idx = list(r_idx)
            # abs_idx = list(np.arange(n_particles+n_shapes, count_nodes))
            # r_idx = np.array(r_idx.extend(abs_idx))
        else:
            r_idx = np.arange(n_particles)
        node_r_idxs.append(r_idx)
        
        if abs_point:
            s_idx = np.arange(count_nodes)
        else:
            s_idx = np.arange(n_particles + n_shapes)
        node_s_idxs.append(s_idx)
        psteps.append(args.pstep)

    if verbose:
        print('clusters', clusters)

    # add heirarchical relations per instance
    cnt_clusters = 0
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]
        n_root_level = len(phases_dict["root_num"][i])

        if n_root_level > 0 and args.hierarchy:
            attr, positions, velocities, count_nodes, \
            rels, node_r_idx, node_s_idx, pstep = \
                    make_hierarchy(args.env, attr, positions, velocities, i, st, ed,
                                   phases_dict, count_nodes, clusters[cnt_clusters], verbose, var)

            for j in range(len(rels)):
                if verbose:
                    print("Relation instance", j, rels[j].shape)
                Rr_idxs.append(torch.LongTensor([rels[j][:, 0], np.arange(rels[j].shape[0])]))
                Rs_idxs.append(torch.LongTensor([rels[j][:, 1], np.arange(rels[j].shape[0])]))
                # relation attr for root-leaf, root-root are 1
                Ra = np.zeros((rels[j].shape[0], args.relation_dim)); Ra[:, 0] = 1
                Ras.append(torch.FloatTensor(Ra))
                values.append(torch.FloatTensor([1] * rels[j].shape[0]))
                node_r_idxs.append(node_r_idx[j])
                node_s_idxs.append(node_s_idx[j])
                psteps.append(pstep[j])

            cnt_clusters += 1

    if verbose:
        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            print("Scene_params:", scene_params)

        print("Attr shape (after hierarchy building):", attr.shape)
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particles], axis=0))
        print("Shape attr:", np.sum(attr[n_particles:n_particles+n_shapes], axis=0))
        print("Roots attr:", np.sum(attr[n_particles+n_shapes:], axis=0))

    ### normalize data
    data = [positions, velocities]
    # stat may store the mean and std value.
    data = normalize(data, stat, var)
    positions, velocities = data[0], data[1]

    if verbose:
        print("Particle positions stats")
        print(positions.shape)
        print(np.min(positions[:n_particles], 0))
        print(np.max(positions[:n_particles], 0))
        print(np.mean(positions[:n_particles], 0))
        print(np.std(positions[:n_particles], 0))

        show_vel_dim = 6 if args.env == 'RiceGrip' else 3
        print("Velocities stats")
        print(velocities.shape)
        print(np.mean(velocities[:n_particles, :show_vel_dim], 0))
        print(np.std(velocities[:n_particles, :show_vel_dim], 0))

    if args.env == 'RiceGrip':
        if var:
            quats = torch.cat(
                [Variable(torch.zeros(n_particles, 4).cuda()), shape_quats,
                 Variable(torch.zeros(count_nodes - n_particles - n_shapes, 4).cuda())], 0)
            state = torch.cat([positions, velocities, quats], 1)
        else:
            quat_null = np.array([[0., 0., 0., 0.]])
            quats = np.repeat(quat_null, [count_nodes], axis=0)
            quats[n_particles:n_particles + n_shapes] = shape_quats
            # if args.eval == 0:
            # quats += np.random.randn(quats.shape[0], 4) * 0.05
            state = torch.FloatTensor(np.concatenate([positions, velocities, quats], axis=1))
    else:
        if var:
            state = torch.cat([positions, velocities], 1)
        else:
            state = torch.FloatTensor(np.concatenate([positions, velocities], axis=1))

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

        for i in range(len(Ras)):
            print(i, np.min(node_r_idxs[i]), np.max(node_r_idxs[i]), np.min(node_s_idxs[i]), np.max(node_s_idxs[i]))

    attr = torch.FloatTensor(attr)
    # Rr_idxs: stage, 2, num_relation. (stage: num of hierachy; relations differ from each other for differenct stage)
    # Rs_idxs: stage, 2, num_relation
    # value: stage, num_relation
    # Ras: stage, num_relation, 1 (leaf-leaf=0, leaf-root root-root=1)
    # node_r_idxs: stage, num_particles (particles are diff for diff hierarchy), only particles idx and root
    # node_s_idxs: stage, num_particles (particles are diff for diff hierarchy), include n_shape's idx
    # psteps: stage
    relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]

    return attr, state, relations, n_particles, n_shapes, instance_idx


def preprocess_transformer(data, stat, env_cfg, phases_dict, verbose, idx_rollout, idx_timestep, label=None):
    attr, state, relations, n_particles, n_shapes, instance_idx = \
            prepare_input(data, stat, env_cfg, phases_dict, verbose, abs_point=False)

    max_particles = attr.shape[0] + env_cfg.num_abs_token
    # label: N_P, dim
    if label is not None:
        if env_cfg.num_abs_token > 0:
            label = torch.cat([label, torch.zeros((max_particles-label.shape[0], label.shape[1]))], dim=0)
        label = DataContainer(label.unsqueeze(0).transpose(-1, -2), stack=True, pad_dims=1)
        
    if env_cfg.num_abs_token > 0:
        attr = torch.cat([attr, torch.zeros((max_particles-attr.shape[0], attr.shape[1]))], dim=0)
        state = torch.cat([state, torch.zeros((max_particles-state.shape[0], state.shape[1]))], dim=0)
    # DataContainer(data, stack=False, padding_value=0, pad_dims=2, cpu_only=False)
    attr = DataContainer(attr.unsqueeze(0).transpose(-1, -2), stack=True, pad_dims=1)
    state = DataContainer(state.unsqueeze(0).transpose(-1, -2), stack=True, pad_dims=1)

    # Label mask
    # 1, N_P
    label_mask = torch.ones(max_particles, requires_grad=False)
    label_mask[n_particles:] = 0
    label_mask = DataContainer(label_mask.unsqueeze(0), stack=True, pad_dims=1)

    # Padding mask
    # 1, N_P
    # pad_mask = torch.zeros(max_particles, requires_grad=False).bool()
    # pad_mask[n_particles+n_shapes:] = True
    # pad_mask = DataContainer(pad_mask.unsqueeze(0), stack=True, padding_value=True, pad_dims=1)

    # relations: relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]

    obj_r_idx, obj_s_idx = relations[0][0], relations[1][0]
    obj_r_idx = obj_r_idx[0]
    obj_s_idx = obj_s_idx[0]
    assert len(obj_r_idx) == len(obj_s_idx)
    assert set(range(n_particles)) == set(obj_r_idx.numpy().tolist())
    # Mask for abs_token
    # if env_cfg.num_abs_token > 0:
    #     for materials_idx, material in enumerate(phases_dict['material']):
    #         particles_start_idx = instance_idx[materials_idx]
    #         particles_end_idx = instance_idx[materials_idx+1]
    #         abs_p = torch.LongTensor(np.ones(particles_end_idx-particles_start_idx, dtype=np.int) * (max_particles - env_cfg.num_abs_token + materials_idx))
    #         nodes = torch.LongTensor(np.arange(particles_end_idx-particles_start_idx) + particles_start_idx)
    #         obj_r_idx = torch.cat((obj_r_idx, abs_p), dim=0)
    #         obj_s_idx = torch.cat((obj_s_idx, nodes), dim=0)
    #         obj_r_idx = torch.cat((obj_r_idx, nodes), dim=0)
    #         obj_s_idx = torch.cat((obj_s_idx, abs_p), dim=0)
    relation_mask = torch.ones((max_particles, max_particles), dtype=torch.long)
    relation_mask.index_put_([obj_r_idx, obj_s_idx], torch.tensor(0))
    # DEBUG: For none mask <<<
    # n_particle_mask = torch.arange(0, n_particles)
    # relation_mask.index_put_([n_particle_mask, n_particle_mask], torch.tensor(0))
    # >>>
    # Add mask for n_shapes
    n_shapes_mask = torch.arange(n_particles, n_particles+n_shapes)
    relation_mask.index_put_([n_shapes_mask, n_shapes_mask], torch.tensor(0))
    num_heads = env_cfg.attn_mask
    # relation_mask = relation_mask.bool().unsqueeze(0).expand(num_heads, -1, -1)
    relation_mask = relation_mask.bool().unsqueeze(0)
    relation_mask = DataContainer(relation_mask, stack=True, padding_value=1, pad_dims=2)

    fluid_mask = torch.zeros(max_particles)
    rigid_mask = torch.zeros(max_particles)
    for i in range(len(instance_idx) - 1):
        st, ed = instance_idx[i], instance_idx[i + 1]
        if phases_dict['material'][i] == 'rigid':
            # mask
            rigid_mask[st:ed] = 1
        elif phases_dict['material'][i] == 'fluid':
            fluid_mask[st:ed] = 1
    fluid_mask = DataContainer(fluid_mask.unsqueeze(0), stack=True, pad_dims=1)
    rigid_mask = DataContainer(rigid_mask.unsqueeze(0), stack=True, pad_dims=1)
    
    # node_r_mask = torch.zeros(max_particles)
    # node_r_mask[:n_particles] = 1
    # node_r_mask = DataContainer(node_r_mask.unsqueeze(0), stack=True, pad_dims=1)

    # Warning: this may be diff from diff version
    stat = torch.FloatTensor(stat)

    meta=dict(idx_rollout=idx_rollout, idx_timestep=idx_timestep, n_particles=n_particles, instance_idx=instance_idx)
    inputs = dict(
        attr=attr,
        state=state,
        fluid_mask=fluid_mask,
        rigid_mask=rigid_mask,
        # pad_mask=pad_mask,
        output_mask=label_mask,
        attn_mask=relation_mask,
        # meta=meta,
        # node_r_mask=node_r_mask,
        stat=stat)
    return inputs, meta, label