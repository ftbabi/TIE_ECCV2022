import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MSEAccuracy(Hook, nn.Module):
    '''
    This is not truly mse. This is same as DPI-Net implement.
    '''

    def __init__(self, reduction='mean'):
        """Module to calculate the accuracy
        """
        super(MSEAccuracy, self).__init__()
        self.sqrt_losses_by_batch = 0.0
        self.batch_num = 0
        self.reduction = reduction
    
    def before_epoch(self, runner):
        self.sqrt_losses_by_batch = 0.0
        self.batch_num = 0


    def forward(self, pred, target, avg_factor=None, reduction_override=None):
        """Forward function to calculate accuracy

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
            avg_factor (torch.Tensor)

        Returns:
            list[float]: The accuracies under different topk criterions.
        """
        reduction = reduction_override if reduction_override else self.reduction

        batch_size, num_particles, output_dim = pred.shape
        self.batch_num += batch_size
        
        with torch.no_grad():
            rst = F.mse_loss(pred, target, reduction='none')
            if reduction == 'none':
                return [rst, rst]
            
            if isinstance(avg_factor, torch.Tensor):
                rst_by_bs = rst.sum((1,2)) / avg_factor
            else:
                rst_by_bs = torch.mean(rst, (1, 2))
            sqrt_rst_by_bs = torch.sqrt(rst_by_bs)
            sum_sqrt_rst = torch.sum(sqrt_rst_by_bs)
            batch_mean_std = sum_sqrt_rst / batch_size

            self.sqrt_losses_by_batch += sum_sqrt_rst
            mean_std = self.sqrt_losses_by_batch / self.batch_num

        return [batch_mean_std, mean_std]
