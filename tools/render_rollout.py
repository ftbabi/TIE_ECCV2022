import os
import os.path as osp
import time
import argparse
import numpy as np
import pickle
import mmcv

from sim.datasets.utils import load_data, calc_box_init_FluidShake
from sim.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the configurations of model')
    parser.add_argument('src_dir', type=str, help='The dir to the simulation/prediction results')
    parser.add_argument('--save_dir', help='The dir to save the rendered simulation/prediction results')

    parser.add_argument('--val_rollout', type=int, default=-1, help='The number of rollout to render')
    parser.add_argument('--tar_rollout', type=int, default=-1, help='Target index of the rollout to render')
    parser.add_argument('--show_only', action='store_true', help='If true, this will not save the rendered image and only display on screen only')

    args = parser.parse_args()
    return args


def get_shape_ground_truth(args, env, rollout_idx):
    position_dim = 3
    shape_state_dim = 14
    if env == 'RiceGrip':
        position_dim = 6
    for step in range(args.time_step - 1):
        data_path = os.path.join(args.data_dir, 'valid', str(rollout_idx), str(step) + '.h5')
        positions, shape_quats = load_data(['positions', 'shape_quats'], data_path)

        if step == 0:
            if env == 'BoxBath':
                n_shapes = 0
                scene_params = np.zeros(1)
            elif env == 'FluidFall':
                n_shapes = 0
                scene_params = np.zeros(1)
            elif env == 'RiceGrip':
                n_shapes = shape_quats.shape[0]
            elif env == 'FluidShake':
                n_shapes = shape_quats.shape[0]
            else:
                raise AssertionError("Unsupported env")

            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            logger.info("n_particles {}".format(n_particles))
            logger.info("n_shapes {}".format(n_shapes))

            p_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, position_dim))
            s_gt = np.zeros((args.time_step - 1, n_shapes, shape_state_dim))

        p_gt[step] = positions[:, -position_dim:]
        if env == 'RiceGrip' or env == 'FluidShake':
            s_gt[step, :, :3] = positions[n_particles:, :3]
            s_gt[step, :, 3:6] = p_gt[max(0, step-1), n_particles:, :3]
            s_gt[step, :, 6:10] = shape_quats
            s_gt[step, :, 10:] = shape_quats
    return s_gt

if __name__ == '__main__':
    args = parse_args()
    mmcv.mkdir_or_exist(osp.abspath(args.save_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.save_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(args)


    logger.info("Loading rollouts from %s" % args.src_dir)
    cfg = mmcv.Config.fromfile(args.config)
    args.time_step = cfg.data.test.env_cfg.time_step
    args.env = cfg.data.test.env_cfg.env
    args.data_dir = cfg.data.test.env_cfg.dataf

    import pyflex
    pyflex.init()

    if args.tar_rollout >= 0:
        infos = [args.tar_rollout]
    elif args.val_rollout > 0:
        infos = np.arange(args.val_rollout)
    else:
        infos = np.arange(cfg.data.test.env_cfg.n_rollout - int(cfg.data.test.env_cfg.n_rollout * cfg.data.test.env_cfg.train_valid_ratio))
    
    if args.env == 'FluidFall':
        env_idx = 4
    elif args.env == 'BoxBath':
        env_idx = 1
    elif args.env == 'FluidShake':
        env_idx = 6
    elif args.env == 'RiceGrip':
        env_idx = 5

    recs = []
    mses = []
    material_mses = []
    vel_losses = []
    time_per_frame_particle = []

    for idx in infos:
        logger.info("Rollout %d / %d" % (idx, len(infos)))
        vel_loss = []
        with open(os.path.join(args.src_dir, "rollout_test_{}.pkl".format(idx)), "rb") as file:
            rollout_data = pickle.load(file)
        p_gt, p_pred = rollout_data['ground_truth_rollout'], rollout_data['predicted_rollout']
        p_gt = p_gt[1:-1]
        frame, particles, dim = p_gt.shape

        time_per_frame = 0
        if 'time_per_frame' in rollout_data.keys():
            time_per_frame = rollout_data['time_per_frame']
        else:
            pass
        time_per_frame_particle.append(time_per_frame / particles)

        mse = (p_gt - p_pred) ** 2
        mean_mse = np.mean(mse)

        particle_type = rollout_data['particle_types']
        fluid_num = np.sum(particle_type)
        n_particles = particle_type.shape[0]
        rigid_num = n_particles - fluid_num
        instance_idx = [0, rigid_num, n_particles]
        material_mse = []
        for i in range(len(instance_idx)-1):
            st, ed = int(instance_idx[i]), int(instance_idx[i+1])
            if ed > st:
                m_mse = mse[:, st:ed, :]
                m_mean_mse = np.mean(m_mse)
                material_mse.append(m_mean_mse)
        logger.info("Separate Materials MSE: {}".format(material_mse))
        mses.append(mean_mse)
        material_mses.append(material_mse)
        logger.info("MSE: {}".format(mean_mse))
        logger.info("Materials MSE: {}".format(np.mean(material_mse)))
        logger.info("Time per frame per particle: {}".format(time_per_frame / particles))
        # Load scene_params
        data_path = os.path.join(args.data_dir, 'valid', str(idx), str(0) + '.h5')
        scene_params = load_data(['scene_params'], data_path)[0]
        s_gt = get_shape_ground_truth(args, args.env, idx)

        initial_offset = 3
        s_gt = s_gt[initial_offset:]
        height = 2.0
        border = 0.025
        des_dir = os.path.join(args.save_dir, 'rollout_%d' % idx)
        os.system('mkdir -p ' + des_dir)
        ##### render for the ground truth
        # if cfg.data.test.env_cfg.env == 'BoxBath':
        #     scene_params = cfg.data.test.env_cfg.scene_params
        pyflex.set_scene(env_idx, scene_params, 0)

        if cfg.data.test.env_cfg.env == 'RiceGrip':
            halfEdge = np.array([0.15, 0.8, 0.15])
            center = np.array([0., 0., 0.])
            quat = np.array([1., 0., 0., 0.])
            pyflex.add_box(halfEdge, center, quat)
            pyflex.add_box(halfEdge, center, quat)
        elif cfg.data.test.env_cfg.env == 'FluidShake':
            x, y, z, dim_x, dim_y, dim_z, box_dis_x, box_dis_z = scene_params

            boxes = calc_box_init_FluidShake(box_dis_x, box_dis_z, height, border)

            x_box = x + (dim_x-1)/2.*0.055

            for box_idx in range(len(boxes) - 1):
                halfEdge = boxes[box_idx][0]
                center = boxes[box_idx][1]
                quat = boxes[box_idx][2]
                pyflex.add_box(halfEdge, center, quat)


        for step in range(min(p_gt.shape[0], s_gt.shape[0])):
            if cfg.data.test.env_cfg.env == 'RiceGrip':
                pyflex.set_shape_states(s_gt[step])
            elif cfg.data.test.env_cfg.env == 'FluidShake':
                pyflex.set_shape_states(s_gt[step, :-1])

            mass = np.zeros((n_particles, 1))
            if cfg.data.test.env_cfg.env == 'RiceGrip':
                p = np.concatenate([p_gt[step, :n_particles, -3:], mass], 1)
            else:
                p = np.concatenate([p_gt[step, :n_particles], mass], 1)

            pyflex.set_positions(p)
            if not args.show_only:
                time.sleep(2/60)
            pyflex.render(capture=args.show_only, path=os.path.join(des_dir, 'gt_%d.tga' % step))

        ##### render for the predictions
        pyflex.set_scene(env_idx, scene_params, 0)

        if cfg.data.test.env_cfg.env == 'RiceGrip':
            pyflex.add_box(halfEdge, center, quat)
            pyflex.add_box(halfEdge, center, quat)
        elif cfg.data.test.env_cfg.env == 'FluidShake':
            for box_idx in range(len(boxes) - 1):
                halfEdge = boxes[box_idx][0]
                center = boxes[box_idx][1]
                quat = boxes[box_idx][2]
                pyflex.add_box(halfEdge, center, quat)

        for step in range(min(p_gt.shape[0], s_gt.shape[0])):
            if cfg.data.test.env_cfg.env == 'RiceGrip':
                pyflex.set_shape_states(s_gt[step])
            elif cfg.data.test.env_cfg.env == 'FluidShake':
                pyflex.set_shape_states(s_gt[step, :-1])

            mass = np.zeros((n_particles, 1))
            if cfg.data.test.env_cfg.env == 'RiceGrip':
                p = np.concatenate([p_pred[step, :n_particles, -3:], mass], 1)
            else:
                p = np.concatenate([p_pred[step, :n_particles], mass], 1)

            pyflex.set_positions(p)
            if not args.show_only:
                time.sleep(2/60)
            pyflex.render(capture=args.show_only, path=os.path.join(des_dir, 'pred_%d.tga' % step))

    pyflex.clean()

    logger.info("Mean pos MSE: {}, std {}".format(np.mean(mses), np.std(mses)))
    # material_mses
    logger.info("Mean pos Materials MSE: {}, std {}".format(np.mean(material_mses), np.std(material_mses)))
    logger.info("Mean pos Separate Materials MSE: {}, std {}".format(np.mean(material_mses, axis=0), np.std(material_mses, axis=0)))
    logger.info("Time per frame per particle MSE: {}, std {}".format(np.mean(time_per_frame_particle), np.std(time_per_frame_particle)))
