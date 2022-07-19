import os
import time
import random
import numpy as np
import tqdm
from torch.utils.data import Dataset

from .builder import DATASETS

from sim.models.losses import MSELoss, MSEAccuracy
from sim.datasets.utils import (load_raw, preprocess_transformer, store_data, load_data, combine_stat, init_stat,
                                normalize, denormalize, gen_PyFleX)


@DATASETS.register_module()
class PhysicsFleXDataset(Dataset):

    def __init__(self, env_cfg, phase, verbose, **kwargs):
        self.env_cfg = env_cfg
        self.phase = phase
        self.phases_dict = self.env_cfg.phases_dict
        self.verbose = verbose
        self.data_dir = os.path.join(self.env_cfg.dataf, phase)
        self.stat_path = os.path.join(self.env_cfg.dataf, 'stat.h5')
        self.cat_with_others = False

        os.system('mkdir -p ' + self.data_dir)

        if self.env_cfg.env == 'RiceGrip':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']
        elif self.env_cfg.env == 'FluidShake':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']
        elif self.env_cfg.env == 'BoxBath':
            self.data_names = ['positions', 'velocities', 'clusters']
        elif self.env_cfg.env == 'FluidFall':
            self.data_names = ['positions', 'velocities']

        ratio = self.env_cfg.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.env_cfg.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.env_cfg.n_rollout - int(self.env_cfg.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")
        
        # if not self.env_cfg.get('gen_meta', False):
        #     self.load_meta(self.env_cfg.env)
        #     if self.env_cfg.env == 'BoxBath' and not self.env_cfg.hierarchy:
        #         # This is important, don't forget
        #         self.max_particles += 5
        #     elif self.env_cfg.env == 'FluidFall' and not self.env_cfg.baseline:
        #         # This is important, don't forget
        #         self.max_particles += 1

        self.walls = None
        if not self.env_cfg.get('gen_data', False):
            self.load_data(self.env_cfg.env)
            if self.env_cfg.env == 'BoxBath' and not self.env_cfg.hierarchy:
                self.load_walls(self.env_cfg.env)
            # if phase == 'valid':
            #     # Only for evaluate()
            #     self.load_annotations(self.env_cfg.env)

    def __len__(self):
        return self.n_rollout * (self.env_cfg.time_step - 1)

    def load_data(self, name):
        self.stat = load_data(self.data_names[:2], self.stat_path)
        for i in range(len(self.stat)):
            self.stat[i] = self.stat[i][-self.env_cfg.position_dim:, :]
            # print(self.data_names[i], self.stat[i].shape)

    def load_meta(self, name):
        label_path = os.path.join(self.data_dir, 'meta.h5')
        # This is n_particles + n_shape
        self.max_particles = load_data(['max_particles'], label_path)[0]
    
    def load_walls(self, name):
        walls_path = os.path.join(self.data_dir, 'walls.h5')    
        self.walls = load_data(['walls'], walls_path)[0]

    def __getitem__(self, idx):
        idx_rollout = idx // (self.env_cfg.time_step - 1)
        idx_timestep = idx % (self.env_cfg.time_step - 1)
        # ignore the first frame for env RiceGrip
        if self.env_cfg.env == 'RiceGrip' and idx_timestep == 0:
            idx_timestep = np.random.randint(1, self.env_cfg.time_step - 1)

        data, label = load_raw(self.data_dir, idx_rollout, idx_timestep, self.data_names, self.env_cfg, self.stat, self.walls, self.cat_with_others)
        inputs, meta, label = self.preprocess(idx_rollout, idx_timestep, data, label=label)

        input_data = dict(inputs=inputs, gt_label=dict(label=label, label_mask=inputs['output_mask']), meta=meta)
        return input_data

    def evaluate(self,
                 results,
                 metric='mse',
                 metric_options={'opt': ['pos', 'vel']},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict): Options for calculating metrics. Allowed
                keys are 'topk', 'thrs' and 'average_mode'.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'mse', 'std'
        ]

        # max_particles is max(n_particles) + n_shapes + n_root
        eval_results = {}
        # Results are [velocities, output_mask, vel_loss, pos_loss, pred_position]
        # batchsize, num_particles, 1
        #
        vel_loss_idx = self.env_cfg.position_dim+1
        pos_loss_idx = self.env_cfg.position_dim+2
        output_mask_idx = self.env_cfg.position_dim#

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        opts = metric_options.get('opt', ['pos'])

        if 'mse' in metrics:
            # Test vel first, which is not a metric in performance but reflect the loss while training
            # The loss has meaned on the last dim, thus final dim==1
            #
            vel_loss_per_frame = []
            pos_loss_per_frame = []
            rollout_length = int(len(results) / self.n_rollout)#
            
            assert rollout_length * self.n_rollout == len(results)#
            vel_loss_per_frame = np.array([np.sum(rst[:, vel_loss_idx] * rst[:, output_mask_idx]) / np.sum(rst[:, output_mask_idx]) for rst in results])#
            pos_loss_per_frame = np.array([np.sum(rst[:, pos_loss_idx] * rst[:, output_mask_idx]) / np.sum(rst[:, output_mask_idx]) for rst in results])#

            vel_sqrt_per_frame = np.sqrt(vel_loss_per_frame)#
            vel_std, vel_mean = np.std(vel_sqrt_per_frame), np.mean(vel_sqrt_per_frame)#

            pos_loss_per_rollout = pos_loss_per_frame.reshape(self.n_rollout, rollout_length)#
            pos_loss_per_rollout = np.mean(pos_loss_per_rollout, axis=-1)#
            pos_std, pos_mean = np.std(pos_loss_per_rollout), np.mean(pos_loss_per_rollout)#
            
            eval_results = {
                'pos mse mean': pos_mean*self.env_cfg.eval_ratio,
                'pos mse std': pos_std*self.env_cfg.eval_ratio,
                'vel mean': vel_mean,
                'vel std': vel_std,
                'eval ratio': self.env_cfg.eval_ratio
            }
        else:
            raise NotImplementedError

        return eval_results

    def gen_meta(self):
        meta_path = os.path.join(self.data_dir, 'meta.h5')
        if os.path.exists(meta_path):
            print("Meta already generated. Please check")
            return 

        print("Generating meta for env {}".format(self.env_cfg.env))
        gt_label_buff = []
        particles_seq = []

        for idx in tqdm.tqdm(range(self.__len__())):
            idx_rollout = idx // (self.env_cfg.time_step - 1)
            idx_timestep = idx % (self.env_cfg.time_step - 1)
            data_nxt_path = os.path.join(self.data_dir, str(idx_rollout), str(idx_timestep + 1) + '.h5')
            data_nxt = normalize(load_data(self.data_names, data_nxt_path), self.stat)
            position = data_nxt[0][:, -self.env_cfg.position_dim:]
            # This include n_shapes, 
            # n_particles is: number of object particles and number of shapes of env's box
            n_particles = position.shape[0]
            particles_seq.append(n_particles)
        
        max_particles = max(particles_seq)
        
        store_data(
            ['max_particles'], 
            [max_particles], 
            meta_path)

    def gen_data(self, num_workers, multi_processing=False):
        infos = []
        assert self.n_rollout % num_workers == 0

        for i in range(num_workers):
            info = {
                'scene_params': self.env_cfg.get('scene_params', None),
                'env': self.env_cfg.env,
                'root_num': self.env_cfg.phases_dict['root_num'],
                'thread_idx': i,
                'data_dir': self.data_dir,
                'data_names': self.data_names,
                'n_rollout': self.n_rollout // num_workers,
                'n_instance': self.env_cfg.n_instance,
                'time_step': self.env_cfg.time_step,
                'time_step_clip': self.env_cfg.time_step_clip,
                'dt': self.env_cfg.dt,
                'shape_state_dim': self.env_cfg.shape_state_dim}

            if self.env_cfg.env == 'BoxBath':
                info['env_idx'] = 1
            elif self.env_cfg.env == 'FluidFall':
                info['env_idx'] = 4
            elif self.env_cfg.env == 'RiceGrip':
                info['env_idx'] = 5
            elif self.env_cfg.env == 'FluidShake':
                info['env_idx'] = 6
            else:
                raise AssertionError("Unsupported env")

            infos.append(info)

        cores = num_workers
        if multi_processing:
            pool = mp.Pool(processes=cores)
            data = pool.map(gen_PyFleX, infos)
        else:
            data = []
            for info in infos:
                d = gen_PyFleX(info)
                data.append(d)

        if self.phase == 'train':
            # positions [x, y, z], velocities[xdot, ydot, zdot]
            if self.env_cfg.env == 'RiceGrip':
                # 6-dim vec for the velocity of the 
                # current observed position and the resting position
                stat = [init_stat(6), init_stat(6)]
            else:
                stat = [init_stat(3), init_stat(3)]
            for i in range(len(data)):
                for j in range(len(stat)):
                    stat[j] = combine_stat(stat[j], data[i][j])
            store_data(self.data_names[:2], stat, self.stat_path)
        
        # self.gen_meta()
    
    def load_data_rollout(self, idx_rollout, frames=-1):
        gt = []
        inputs = []
        if frames <= 0:
            frames = self.env_cfg.time_step-1
        for i in range(frames):
            if self.env_cfg.env == 'BoxBath':
                data, label = load_raw(self.data_dir, idx_rollout, i, self.data_names, self.env_cfg, self.stat, self.walls, self.cat_with_others)
            else:
                data, label = load_raw(self.data_dir, idx_rollout, i, self.data_names, self.env_cfg, self.stat)
            gt.append(data.copy())
            data, meta, _ = self.preprocess(idx_rollout, i, data)
            inputs.append(data.copy())

        return gt, inputs
    
    def preprocess(self, idx_rollout, idx_timestep, data, label=None):
        data, meta, label = preprocess_transformer(data, self.stat, self.env_cfg, self.phases_dict, self.verbose, idx_rollout, idx_timestep, label=label)
        return data, meta, label
