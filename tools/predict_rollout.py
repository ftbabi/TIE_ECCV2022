import imp
import os
import os.path as osp
import time
import argparse
import numpy as np
import pickle
from sim.datasets import build_dataset

import torch
import torch.nn.functional as F
import pickle
from sim.datasets.utils import denormalize
from sim.utils import get_root_logger, count_parameters
from sim.datasets.utils import to_tensor_cuda
from mmcv.parallel import collate, scatter

import mmcv
from mmcv.runner import load_checkpoint
from sim.models import build_simulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default=None)
    parser.add_argument('work_dir', help='the dir to save logs and models')
    parser.add_argument('--val_rollout', type=int, default=-1)
    parser.add_argument('--tar_rollout', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # check work dir
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(args)
    assert torch.cuda.is_available()

    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    model = build_simulator(cfg.model)
    logger.info("Number of parameters: %d" % count_parameters(model))
    logger.info("Loading network from %s" % args.checkpoint)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    criterionMSE = F.mse_loss
    model.to(args.device)
    model.eval()

    if args.tar_rollout > 0:
        infos = [args.tar_rollout]
    elif args.val_rollout > 0:
        infos = np.arange(args.val_rollout)
    else:
        infos = np.arange(cfg.data.test.env_cfg.n_rollout - int(cfg.data.test.env_cfg.n_rollout * cfg.data.test.env_cfg.train_valid_ratio))

    recs = []
    mses = []
    material_mses = []
    time_per_frame_list = []

    for idx in infos:

        logger.info("Rollout %d / %d" % (idx, len(infos)))
        vel_loss = []
        frames_time = []

        gt_all, _ = dataset.load_data_rollout(idx)
        p_gt = np.stack([i[0][:, -cfg.data.test.env_cfg.position_dim:] for i in gt_all])
        v_nxt_gt = np.stack([i[1][:, -cfg.data.test.env_cfg.position_dim:] for i in gt_all])
        p_pred = []

        start_frame = 1
        inputs_raw = gt_all[start_frame]

        # Omit the RiceGrip 0th frame. Cuz the states are 0s.
        for step in range(start_frame, cfg.data.test.env_cfg.time_step - 2):
            if step % 10 == 0:
                logger.info("Step %d / %d: time per frame %.6f" % (step, cfg.data.test.env_cfg.time_step - 1, np.sum(frames_time) / len(frames_time)))

            # logger.info('Time prepare input', time.time() - st_time)
            # st_time = time.time()
            inputs, meta, _ = dataset.preprocess(idx, step, inputs_raw)
            stat = inputs['stat'].data.cpu().numpy()
            inputs = collate([inputs], samples_per_gpu=1)
            inputs = scatter(inputs, [next(model.parameters()).device.index])[0]
            n_particles = meta['n_particles']
            instance_idx = meta['instance_idx']
            with torch.no_grad():
                st_time = time.time()
                vels = model(inputs=inputs, return_loss=False)
                ed_time = time.time()
                vels = torch.from_numpy(vels[0][:n_particles, :cfg.data.test.env_cfg.position_dim])
            frames_time.append(ed_time-st_time)

            if 'LossHead' != cfg.model.head.type:
                # No denormalize for DLF model
                vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]
            else:
                vels = vels.data.cpu().numpy()

            # Fill the n_shape ground truth
            vels = np.concatenate([vels, v_nxt_gt[step+1, n_particles:]], 0)
            inputs_raw[0] = inputs_raw[0] + vels * cfg.data.test.env_cfg.dt

            if cfg.data.test.env_cfg.env == 'RiceGrip':
                # shifting the history
                # positions, restPositions
                inputs_raw[1][:, cfg.data.test.env_cfg.position_dim:] = inputs_raw[1][:, :-cfg.data.test.env_cfg.position_dim]
            inputs_raw[1][:, :cfg.data.test.env_cfg.position_dim] = vels
            p_pred.append(inputs_raw[0].copy())

            # # This means, always predict t+1 frame by the gt_label at t frame.
            if args.debug:
                inputs_raw[0] = p_gt[step + 1].copy()
                inputs_raw[1][:, :cfg.data.test.env_cfg.position_dim] = v_nxt_gt[step]
        
        cur_time_per_frame = np.sum(frames_time) / len(frames_time)
        time_per_frame_list.append(cur_time_per_frame)
        # Cal mse for positions
        gt_labl = torch.from_numpy(p_gt[start_frame+1:])
        # For ricegrip, only the particle pos
        gt_labl = gt_labl[:, :, -3:]
        p_pred = np.stack(p_pred)
        pred_pos = torch.from_numpy(p_pred)
        pred_pos = pred_pos[:, :, -3:]
        with torch.no_grad():
            loss = criterionMSE(pred_pos, gt_labl, reduction='none')
            avg_factor = pred_pos.shape[0]*pred_pos.shape[1]*pred_pos.shape[2]
            mse_single = loss.sum() / avg_factor
            mses.append(mse_single)

            # Materials MSE
            material_mse = []
            for i in range(len(instance_idx)-1):
                st, ed = instance_idx[i], instance_idx[i+1]
                material_loss = loss[:, st:ed, :]
                avg_factor = pred_pos.shape[0]*(ed-st)*pred_pos.shape[2]
                material_loss = material_loss.sum() / avg_factor
                material_mse.append(material_loss)
            mean_m_mse = np.mean(material_mse)
            material_mses.append(mean_m_mse)
            logger.info("MSE: {}".format(mse_single.item()))
            logger.info("Materials MSE: {}".format(mean_m_mse))
            logger.info("Time per frame: {}".format(cur_time_per_frame))
        # Save pickle
        particle_types = np.ones(n_particles)
        if cfg.data.test.env_cfg.env == 'BoxBath':
            rigid_num = instance_idx[1]
            particle_types[:rigid_num] = 0
        save_path = os.path.join(args.work_dir, 'rollout_test_{}.pkl'.format(idx))
        save_data = {
            'ground_truth_rollout': p_gt[2:],
            'predicted_rollout': p_pred[1:-1],
            'particle_types': particle_types,
            'time_per_frame': cur_time_per_frame,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

    mses = torch.stack(mses, dim=0)
    logger.info("Mean pos MSE: {}, std {}".format(torch.mean(mses).item(), torch.std(mses).item()))
    logger.info("Mean pos Materials MSE: {}, std {}".format(np.mean(material_mses), np.std(material_mses)))
    logger.info("Mean forward time per frame: {}, std {}".format(np.mean(time_per_frame_list), np.std(time_per_frame_list)))
