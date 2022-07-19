import argparse
from pickle import encode_long
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os
import logging
import scipy.spatial as spatial

from mmcv.utils import get_logger

import torch

def to_tensor_cuda(data):
    cuda_data = dict()
    for key, val in data.items():
        if isinstance(val, list):
            cuda_list = [i.unsqueeze(0).cuda() for i in val]
            cuda_data[key] = cuda_list
        else:
            cuda_data[key] = val.unsqueeze(0).cuda()
    
    return cuda_data

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.

        stat_dim = stat[i].shape[0]
        n_rep = int(data[i].shape[1] / stat_dim)
        data[i] = data[i].reshape((-1, n_rep, stat_dim))

        data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

        data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, mask=None, val=False):
    for i in range(len(stat)):
        if isinstance(data[i], torch.Tensor):
            # data[i]: bs, n_particles, 3
            # stat[i]: bs, 3, 2
            data[i] = data[i] * (stat[i][:, :, 1].unsqueeze(1)) + stat[i][:, :, 0].unsqueeze(1)
            # mask: bs, n_particles
            assert isinstance(mask, torch.Tensor)
            data[i] = data[i] * mask.unsqueeze(2)
        elif isinstance(data[i], np.ndarray):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
            if mask is not None:
                assert isinstance(mask, np.ndarray)
                assert data[i].shape[0] == mask.shape[0]
                data[i] = data[i] * mask

    return data


def rotateByQuat(p, quat):
    R = np.zeros((3, 3))
    a, b, c, d = quat[3], quat[0], quat[1], quat[2]
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * b * c - 2 * a * d
    R[0, 2] = 2 * b * d + 2 * a * c
    R[1, 0] = 2 * b * c + 2 * a * d
    R[1, 1] = a**2 - b**2 + c**2 - d**2
    R[1, 2] = 2 * c * d - 2 * a * b
    R[2, 0] = 2 * b * d - 2 * a * c
    R[2, 1] = 2 * c * d + 2 * a * b
    R[2, 2] = a**2 - b**2 - c**2 + d**2

    return np.dot(R, p)


def find_relations_neighbor(positions, query_idx, anchor_idx, radius, order, var=False):
    """
    Find neighbors. For given points indexed by anchor_idx, find 
        those anchors which are close to the query particles. That is,
        find the anchors close to querys.

    Return:
        relations(list):
            [receiver_idx(int), sender_idx(int), relation_type(int)]
    """
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender, np.ones(count_neighbors)], axis=1))

    return relations


def make_hierarchy(env, attr, positions, velocities, idx, st, ed, phases_dict, count_nodes, clusters, verbose=0, var=False, add_offset=True):
    order = 2
    n_root_level = len(phases_dict["root_num"][idx])
    attr, relations, node_r_idx, node_s_idx, pstep = [attr], [], [], [], []

    relations_rev, node_r_idx_rev, node_s_idx_rev, pstep_rev = [], [], [], []

    pos = positions.data.cpu().numpy() if var else positions
    vel = velocities.data.cpu().numpy() if var else velocities

    for i in range(n_root_level):
        root_num = phases_dict["root_num"][idx][i]
        root_sib_radius = phases_dict["root_sib_radius"][idx][i]
        root_des_radius = phases_dict["root_des_radius"][idx][i]
        root_pstep = phases_dict["root_pstep"][idx][i]

        if verbose:
            print('root info', root_num, root_sib_radius, root_des_radius, root_pstep)


        ### clustring the nodes
        # st_time = time.time()
        # kmeans = MiniBatchKMeans(n_clusters=root_num, random_state=0).fit(pos[st:ed, 3:6])
        # print('Time on kmeans', time.time() - st_time)
        # clusters = kmeans.labels_
        # centers = kmeans.cluster_centers_


        ### relations between roots and desendants
        rels, rels_rev = [], []
        # particles to root
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(st, ed))
        # root to particles
        node_r_idx_rev.append(node_s_idx[-1])
        node_s_idx_rev.append(node_r_idx[-1])
        pstep.append(1); pstep_rev.append(1)

        if verbose:
            centers = np.zeros((root_num, 3))
            for j in range(root_num):
                des = np.nonzero(clusters[i][0]==j)[0]
                center = np.mean(pos[st:ed][des, -3:], 0, keepdims=True)
                centers[j] = center[0]
                visualize_neighbors(pos[st:ed], center, 0, des)

        for j in range(root_num):
            desendants = np.nonzero(clusters[i][0]==j)[0]
            if add_offset:
                roots = np.ones(desendants.shape[0]) * j + count_nodes
            else:
                roots = np.ones(desendants.shape[0]) * j
            if verbose:
                print(roots, desendants)
            rels += [np.stack([roots, desendants, np.zeros(desendants.shape[0])], axis=1)]
            rels_rev += [np.stack([desendants, roots, np.zeros(desendants.shape[0])], axis=1)]
            if verbose:
                print(np.max(np.sqrt(np.sum(np.square(pos[st + desendants, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))
        relations_rev.append(np.concatenate(rels_rev, 0))


        ### relations between roots and roots
        # point_tree = spatial.cKDTree(centers)
        # neighbors = point_tree.query_ball_point(centers, root_sib_radius, p=order)

        '''
        for j in range(len(neighbors)):
            visualize_neighbors(centers, centers, j, neighbors[j])
        '''

        # Root to root
        rels = []
        node_r_idx.append(np.arange(count_nodes, count_nodes + root_num))
        node_s_idx.append(np.arange(count_nodes, count_nodes + root_num))
        pstep.append(root_pstep)

        # repeat each element in np.arange(root_num) for root_num times one by one. eg. [0,0,0,1,1,1,...]
        roots = np.repeat(np.arange(root_num), root_num)
        if add_offset:
            roots += count_nodes
        # cat np.arange(root_num) root_num times, eg: [0,1,2,0,1,2,0,1,2]
        siblings = np.tile(np.arange(root_num), root_num)
        if add_offset:
            siblings += count_nodes

        if verbose:
            print(roots, siblings)
        rels += [np.stack([roots, siblings, np.zeros(root_num * root_num)], axis=1)]
        if verbose:
            print(np.max(np.sqrt(np.sum(np.square(centers[siblings, :3] - centers[j]), 1))))

        relations.append(np.concatenate(rels, 0))


        ### add to attributes/positions/velocities
        positions = [positions]
        velocities = [velocities]
        attributes = []
        for j in range(root_num):
            ids = np.nonzero(clusters[i][0]==j)[0]
            if var:
                positions += [torch.mean(positions[0][st:ed, :][ids], 0, keepdim=True)]
                velocities += [torch.mean(velocities[0][st:ed, :][ids], 0, keepdim=True)]
            else:
                # use the avg value of particles in the same cluster as the root attr
                positions += [np.mean(positions[0][st:ed, :][ids], 0, keepdims=True)]
                velocities += [np.mean(velocities[0][st:ed, :][ids], 0, keepdims=True)]

            attributes += [np.mean(attr[0][st:ed, :][ids], 0, keepdims=True)]

        attributes = np.concatenate(attributes, 0)

        if env == 'BoxBath':
            attributes[:, 2 + i] = 1
        elif env == 'RiceGrip':
            attributes[:, 1 + i] = 1

        if verbose:
            print('Attr sum', np.sum(attributes, 0))

        attr += [attributes]
        if var:
            positions = torch.cat(positions, 0)
            velocities = torch.cat(velocities, 0)
        else:
            positions = np.concatenate(positions, 0)
            velocities = np.concatenate(velocities, 0)

        st = count_nodes
        ed = count_nodes + root_num
        count_nodes += root_num

        if verbose:
            print(st, ed, count_nodes, positions.shape, velocities.shape)

    attr = np.concatenate(attr, 0)
    if verbose:
        print("attr", attr.shape)

    relations += relations_rev[::-1]
    node_r_idx += node_r_idx_rev[::-1]
    node_s_idx += node_s_idx_rev[::-1]
    pstep += pstep_rev[::-1]

    return attr, positions, velocities, count_nodes, relations, node_r_idx, node_s_idx, pstep
