import torch.nn as nn

from ..builder import SIMULATORS, build_backbone, build_head, build_neck
from .base import BaseSimulator


@SIMULATORS.register_module()
class ParticleSimulator(BaseSimulator):

    def __init__(self, backbone, neck=None, head=None, pretrained=None, residual=1.0/60.0):
        super(ParticleSimulator, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)
        
        self.residual = residual

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(ParticleSimulator, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, inputs, out_idxs=None):
        """Directly extract features from the backbone + neck
        """
        if out_idxs is not None:
            x, out_slices = self.backbone(**inputs)
        else:
            x = self.backbone(**inputs)
        if self.with_neck:
            x = self.neck(x)
        if out_idxs:
            return x, out_slices
        else:
            return x

    def forward_train(self, inputs, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            inputs (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(inputs)

        losses = dict()
        loss = self.head.forward_train(x, gt_label, inputs=inputs)
        losses.update(loss)

        return losses

    def simple_test(self, inputs, **kwargs):
        """Test without augmentation.

            kwargs: gt_labels, which is not used currently
        """
        out_idxs = inputs.get('out_idxs', None)
        if out_idxs is not None:
            x, out_slices = self.extract_feat(inputs, out_idxs=out_idxs)
        else:
            x = self.extract_feat(inputs, out_idxs=out_idxs)
        if out_idxs is not None:
            return self.head.simple_test(x, inputs=inputs, **kwargs), out_slices
        else:
            return self.head.simple_test(x, inputs=inputs, **kwargs)
