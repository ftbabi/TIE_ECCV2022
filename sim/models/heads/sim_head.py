import torch
import torch.nn.functional as F

from sim.models.losses import MSEAccuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class SimHead(BaseHead):
    """Simulation head.

    Args:
        loss (dict): Config of loss.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='MSELoss', loss_weight=1.0)):
        super(SimHead, self).__init__()

        assert isinstance(loss, dict)

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = MSEAccuracy()

    def loss(self, vel_dist, gt_label, reduction_override=None, weight=None, **kwargs):
        inputs = kwargs.get('inputs', None)
        assert isinstance(inputs, dict)

        n_particles = vel_dist.shape[0] * vel_dist.shape[1]
        acc_factor = n_particles

        output_mask = inputs.get('output_mask', None)
        pad_mask = inputs.get('pad_mask', None)
        if isinstance(output_mask, torch.Tensor):
            output_mask = output_mask.squeeze(1)
            acc_factor = torch.sum(output_mask.int(), dim=1)
            n_particles = torch.sum(acc_factor)
        elif isinstance(pad_mask, torch.Tensor):
            pad_mask = pad_mask.squeeze(1)
            acc_factor = torch.sum((~pad_mask).int(), dim=1)
            n_particles = torch.sum(acc_factor)

        losses = dict()
        # compute loss
        loss = self.compute_loss(vel_dist, gt_label, weight=weight, avg_factor=n_particles * vel_dist.shape[2], reduction_override=reduction_override)
        # compute accuracy
        acc = self.compute_accuracy(vel_dist, gt_label, avg_factor=acc_factor * vel_dist.shape[2], reduction_override=reduction_override)
        losses['loss'] = loss
        losses['accuracy'] = {
            'Batch std': acc[0], 'Agg std': acc[1]}
        return losses

    def forward_train(self, pred, gt_label):
        losses = self.loss(pred, gt_label)
        return losses

    def simple_test(self, pred):
        """Test without augmentation."""
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
