import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .sim_head import SimHead
from sim.datasets.utils import denormalize
from sim.models.utils import FFN


@HEADS.register_module()
class ParticleHead(SimHead):
    """Particle based simulation head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 out_channels=3,
                 in_channels=0,
                 loss=dict(type='MSELoss', loss_weight=1.0),
                 dt=1/60.0,
                 seperate=True,
                 rotation_dim=4,
                 weighted=False,
                 eps=1e-7):
        super(ParticleHead, self).__init__(loss=loss)
        # nf_effect
        self.in_channels = in_channels
        # position_dim
        self.out_channels = out_channels
        self.dt = dt
        self.rotation_dim = rotation_dim
        self.weighted = weighted
        self.seperate = seperate
        self.eps = eps

        if self.out_channels <= 0:
            raise ValueError(
                f'out_channels={out_channels} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        out_channels = self.out_channels + self.rotation_dim
        
        if self.seperate:
            self.rigid_predictor = FFN([self.in_channels, self.in_channels, self.in_channels, out_channels])
            out_channels = self.out_channels
        self.fluid_predictor = FFN([self.in_channels, self.in_channels, self.in_channels, out_channels])
    
    def evaluate(self, vel_dist, inputs, gt_label):
        # bs, n_particles, state_dim
        gt_label = gt_label['label'].squeeze(1).transpose(-1, -2)
        # bs, n_particles, state_dim
        prev_state = inputs['state'].squeeze(1).transpose(-1, -2)
        # bs, n_particles
        output_mask = inputs['output_mask'].squeeze(1)
        # bs, 2, 3, 3
        stat = inputs['stat']
        # bs, 3
        mean_p = stat[:, 0, :, 0]
        std_p = stat[:, 0, :, 1]
        mean_v = stat[:, 1, :, 0]
        std_v = stat[:, 1, :, 1]
        # bs, 3, 2
        pos_stat = torch.stack([mean_p, std_p], dim=2)
        vel_stat = torch.stack([mean_v, std_v], dim=2)

        state_dim = prev_state.shape[-1]
        valid_pos_dim = self.out_channels
        if state_dim == 2*self.out_channels:
            prev_pos = prev_state[:, :, :self.out_channels]
            prev_vel = prev_state[:, :, self.out_channels:]
            vel_label = gt_label[:, :, self.out_channels:]
            pos_label = gt_label[:, :, :self.out_channels]
        else:
            assert self.out_channels % 2 == 0
            valid_pos_dim = int(self.out_channels/2)
            prev_pos = prev_state[:, :, valid_pos_dim:self.out_channels]
            pos_label = gt_label[:, :, valid_pos_dim:self.out_channels]

            prev_vel = prev_state[:, :, self.out_channels+valid_pos_dim:2*self.out_channels]
            # compute loss first
            vel_label = gt_label[:, :, self.out_channels:]
            # prev: bs, 6, 2
            # after: bs, 3, 2
            vel_stat = vel_stat[:, :valid_pos_dim, :]
            pos_stat = pos_stat[:, :valid_pos_dim, :]
        
        vel_loss = self.loss(vel_dist, vel_label, inputs=inputs, reduction_override='none')
        
        if state_dim != 2*self.out_channels:
            # Ricegrip
            vel_label = vel_label[:, :, valid_pos_dim:]
            vel_dist = vel_dist[:, :, valid_pos_dim:]

        prev_pos, vel_dist = denormalize(
            [prev_pos, vel_dist], [pos_stat, vel_stat], mask=output_mask)
        pos_results = vel_dist * self.dt + prev_pos
        pos_label = denormalize([pos_label], [pos_stat], mask=output_mask)[0]
        pos_loss = self.loss(pos_results, pos_label, inputs=inputs, reduction_override='none')

        # Parse loss
        # bs, n_particles, 3/6
        vel_loss = vel_loss['accuracy']['Batch std']
        # bs, n_particles, 3
        pos_loss = pos_loss['loss']
        return vel_loss, pos_loss, pos_results

    def simple_test(self, particle_emb, gt_label=None, **kwargs):
        """Test without augmentation."""
        inputs = kwargs.get('inputs', None)
        assert isinstance(inputs, dict)
        vel_dist = self.predict(particle_emb, inputs)
        # bs, n_particles, 1
        output_mask = inputs['output_mask'].transpose(-1, -2)

        rst = torch.cat([vel_dist, output_mask], dim=2)

        if gt_label is not None:
            # bs, n_particles, 3
            vel_loss, pos_loss, pred_pos = self.evaluate(vel_dist, inputs, gt_label)
            # bs, n_particles, 1
            vel_loss = vel_loss.mean(dim=2, keepdim=True)
            pos_loss = pos_loss.mean(dim=2, keepdim=True)
            rst = torch.cat([rst, vel_loss, pos_loss, pred_pos], dim=2)
        if torch.onnx.is_in_onnx_export():
            return rst
        rst = list(rst.detach().cpu().numpy())
        return rst

    def forward_train(self, particle_emb, gt_label, **kwargs):
        inputs = kwargs.get('inputs', None)
        assert isinstance(inputs, dict)
        vel_dist = self.predict(particle_emb, inputs)

        label = gt_label['label'].squeeze(1).transpose(-1, -2)
        label_mask = inputs['output_mask'].squeeze(1)
        rigid_mask = inputs['rigid_mask'].squeeze(1)
        fluid_mask = inputs['fluid_mask'].squeeze(1)
        
        vel_dist = vel_dist * label_mask.unsqueeze(2)
        # bs, nparticles, [pos, vel]; vel in 3
        vel_label = label[:, :, self.out_channels:] * label_mask.unsqueeze(2)
        weight = None
        if self.weighted:
            weight_factor = fluid_mask.sum(dim=-1, keepdim=True) / (rigid_mask.sum(dim=-1, keepdim=True) + self.eps)
            weight = rigid_mask * weight_factor + fluid_mask
            weight = weight.unsqueeze(-1)
        losses = self.loss(vel_dist, vel_label, weight=weight, inputs=inputs)
        return losses

    def rotation_matrix_from_quaternion(self, params):
        # params dim - 4: bs, (w, x, y, z)

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / (torch.norm(params, dim=1, keepdim=True) + self.eps)
        # bs, 1
        w, x, y, z = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        one = torch.ones_like(w)

        rot = torch.stack((
            torch.stack((1-y*y*2-z*z*2, x*y*2+z*w*2, x*z*2-y*w*2), dim=1),
            torch.stack((x*y*2-z*w*2, 1-x*x*2-z*z*2, y*z*2+x*w*2), dim=1),
            torch.stack((x*z*2+y*w*2, y*z*2-x*w*2, 1-x*x*2-y*y*2), dim=1)), dim=1)

        return rot

    def predict(self, particle_emb, inputs):
        # Init inputs
        state = inputs['state'].squeeze(1).transpose(-1, -2)
        rigid_mask = inputs['rigid_mask'].squeeze(1)
        fluid_mask = inputs['fluid_mask'].squeeze(1)
        # bs, 2, 3, 3
        stat = inputs['stat']
        # bs, 3
        mean_p = stat[:, 0, :, 0]
        std_p = stat[:, 0, :, 1]
        mean_v = stat[:, 1, :, 0]
        std_v = stat[:, 1, :, 1]

        # Predict
        rigid_mask = rigid_mask.unsqueeze(2)
        fluid_mask = fluid_mask.unsqueeze(2)
        rigid_particles = particle_emb * rigid_mask
        # fluid_particles = particle_emb * fluid_mask
        # Pred fluid
        pred_fluid = self.fluid_predictor(particle_emb)

        ## Rigid
        if torch.sum(rigid_mask) > 0:
            if self.seperate:
                pred_rigid = self.rigid_predictor(rigid_particles)
            else:
                pred_rigid = pred_fluid

            if self.rotation_dim > 0:
                # bs, 7
                pred_rigid = torch.sum(pred_rigid * rigid_mask, dim=1)/(torch.sum(rigid_mask, dim=1)+self.eps)
                # R: bs, 3, 3
                R = self.rotation_matrix_from_quaternion(pred_rigid[:, :4])
                b = (pred_rigid[:, 4:] * std_v + mean_v) * self.dt
                # bs, n_p, 3
                p_0 = (state[:, :, :3] * std_p.unsqueeze(1) + mean_p.unsqueeze(1)) * rigid_mask
                # bs, 3
                c = torch.sum(p_0, dim=1) / (torch.sum(rigid_mask, dim=1) + self.eps)
                # bs, n_p, 3   *   bs, 3, 3 = bs, n_p, 3
                p_1 = torch.bmm(p_0 - c.unsqueeze(1), R) + b.unsqueeze(1) + c.unsqueeze(1)
                v = (p_1 - p_0) / self.dt
                pred_rigid = (v-mean_v.unsqueeze(1)) / std_v.unsqueeze(1)

                pred_fluid = pred_fluid[:, :, -self.out_channels:]

            pred = pred_rigid * rigid_mask + pred_fluid * fluid_mask
        else:
            if self.rotation_dim > 0:
                pred_fluid = pred_fluid[:, :, -self.out_channels:]
            pred = pred_fluid * fluid_mask
        
        return pred
