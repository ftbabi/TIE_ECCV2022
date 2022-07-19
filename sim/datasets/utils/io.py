import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py

from .misc import normalize


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data

def load_raw(data_dir, idx_rollout, idx_timestep, data_names, env_cfg, stat, walls=None, cat_with_others=False):
    data_path = os.path.join(data_dir, str(idx_rollout), str(idx_timestep) + '.h5')

    data = load_data(data_names, data_path)
    if env_cfg.env == 'BoxBath':
        wall_pos, wall_vel = walls
        x_center = (wall_pos[1][0]+wall_pos[2][0])/2
        z_center = (wall_pos[3][2]+wall_pos[4][2])/2
        y_center = wall_pos[0][1] + 0.77/2
        if cat_with_others:
            wall_pos[0] += np.array([x_center, 0, z_center])
            wall_pos[1] += np.array([0, y_center, z_center])
            wall_pos[2] += np.array([0, y_center, z_center])
            wall_pos[3] += np.array([x_center, y_center, 0])
            wall_pos[4] += np.array([x_center, y_center, 0])

        data[0] = np.concatenate((data[0], wall_pos), axis=0)
        data[1] = np.concatenate((data[1], wall_vel), axis=0)
    if env_cfg.env == 'FluidFall':
        wall_pos = np.zeros((1, data[0].shape[1]))
        wall_vel = np.zeros((1, data[1].shape[1]))
        data[0] = np.concatenate((data[0], wall_pos), axis=0)
        data[1] = np.concatenate((data[1], wall_vel), axis=0)

    vel_his = []
    for i in range(env_cfg.n_his):
        # Note: Original: str(max(1, idx_timestep - i - 1))
        path = os.path.join(data_dir, str(idx_rollout), str(max(0, idx_timestep - i - 1)) + '.h5')
        data_his = load_data(data_names, path)
        vel_his.append(data_his[1])

    data[1] = np.concatenate([data[1]] + vel_his, 1)

    ### label
    # The behavior in this and in gen_label are different, need further check 
    data_nxt_path = os.path.join(data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')
    data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
    position = torch.FloatTensor(data_nxt[0][:, -env_cfg.position_dim:])
    velocity = torch.FloatTensor(data_nxt[1])
    label = torch.cat([position, velocity], dim=1)
    # The order is: particles, env's shape, root
    # Only use n_particles is ok

    if env_cfg.env == 'BoxBath' or env_cfg.env == 'FluidFall':
        wall_pos = torch.FloatTensor(wall_pos)
        wall_vel = torch.FloatTensor(wall_vel)
        label_wall = torch.cat([wall_pos, wall_vel], dim=1)
        label = torch.cat([label, label_wall], dim=0)
        
    return data, label