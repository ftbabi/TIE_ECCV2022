import torch
import torch.nn as nn
import math

from mmcv.cnn import build_activation_layer, build_norm_layer
from sim.models.utils import FFN

from ..builder import ATTENTION
from .initialization import kaiming_uniform_, kaiming_normal_
from .norm_func import mean_std_attn


@ATTENTION.register_module()
class AttentionTIE(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-5,
                 s_attn=True, s_mean=True, s_std=True, **kwargs):
        super(AttentionTIE, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.eps = eps
        self.s_attn = s_attn
        self.s_mean = s_mean
        self.s_std = s_std

        self.dropout = nn.Dropout(dropout)
        fan_in = 4 * dim
        
        self.relation_attn = nn.Parameter(torch.Tensor(dim, 3 * dim))
        nn.init.kaiming_uniform_(self.relation_attn, a=math.sqrt(5))
        fan_in = 3 * dim
        
        self.q_proj = FFN([dim, dim], final_act=False, bias=False, act_cfg=act_cfg)
        self.norms = nn.ModuleList()
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.r_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.s_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)

        self.eps = eps
        if self.s_mean or self.s_std:
            self.n_weight = nn.Parameter(torch.Tensor(1, 1, dim))
            self.n_bias = nn.Parameter(torch.Tensor(1, 1, dim))
            nn.init.ones_(self.n_weight)
            nn.init.zeros_(self.n_bias)
        else:
            self.n_weight = torch.ones(1, 1, dim)
            self.n_bias = torch.zeros(1, 1, dim)

    def forward(self, x, attn_mask, key_padding_mask=None, output_mask=None, receiver_val_res=None, sender_val_res=None, residual_receiver=None, residual_sender=None, node_r_mask=None, **kwargs):
        '''
        Args:
            x: n_particles, bs, embed_dim
            attn_mask: bs, headnum, n_particles, n_particles
            receiver_val_res: bs, n_particles, embed_dim Need norm before into this!!!
            receiver_val_res: n_particles, bs, embed_dim
            key_padding_mask: bs, num_key,
        '''
        x = x.transpose(0, 1)
        B, N, C = x.shape

        # B, N, C
        receiver_val_res = receiver_val_res.transpose(0, 1)
        sender_val_res = sender_val_res.transpose(0, 1)
        residual_receiver = residual_receiver.transpose(0, 1)
        residual_sender = residual_sender.transpose(0, 1)

        receiver_weight = self.relation_attn[:, self.dim:2*self.dim]
        sender_weight = self.relation_attn[:, 2*self.dim:]
        memory_weight = self.relation_attn[:, :self.dim]
        memory_r = receiver_val_res.matmul(memory_weight.t())
        memory_s = sender_val_res.matmul(memory_weight.t())
        v_receiver = x.matmul(receiver_weight.t()) + memory_r
        v_sender = x.matmul(sender_weight.t()) + memory_s
        
        v_receiver += residual_receiver
        v_sender += residual_sender

        # After: B, num_heads, N, C//num_heads
        v_receiver = v_receiver.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_sender = v_sender.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q: B, num_heads, N, C//num_heads
        # No trainable parameter
        mean_rs, std_rs, attn = mean_std_attn(q, v_receiver, v_sender, attn_mask, eps=self.eps)
        if not self.s_attn:
            attn = attn.masked_fill(~attn_mask, 1.0)
        if not self.s_mean:
            mean_rs = torch.zeros_like(mean_rs).cuda()
        if not self.s_std:
            std_rs = torch.ones_like(std_rs).cuda()
        attn = (attn - torch.sum(q, dim=-1, keepdim=True) * mean_rs) / std_rs

        attn = attn * self.scale
        attn.masked_fill_(attn_mask, float('-inf'))
        if key_padding_mask is not None:
            attn.masked_fill_(key_padding_mask.unsqueeze(1), float('-inf'))

        # attn B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = attn.masked_fill(attn_mask, 0.0)
        # attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
        attn = self.dropout(attn)

        attn_std = attn / std_rs
        mean_rs = mean_rs * attn_std

        x = (torch.matmul(attn_std, v_sender) - torch.sum(mean_rs, dim=-1, keepdim=True))
        
        # bs, num_heads, N, head_dim
        x = x + torch.sum(attn_std, dim=-1, keepdim=True) * v_receiver
        x = (x.transpose(1, 2).reshape(B, N, C))
        x = x * self.n_weight.reshape(1,1,-1) + self.n_bias.reshape(1,1,-1)
        if node_r_mask is not None:
            x = x * node_r_mask.bool().unsqueeze(2)
        v_receiver = v_receiver.transpose(1, 2).reshape(B, N, C)
        v_sender = v_sender.transpose(1, 2).reshape(B, N, C)
        # >>>

        x = self.norms[0](x)
        v_receiver = self.norms[1](v_receiver)
        v_sender = self.norms[2](v_sender)
        x = self.proj(x)
        v_receiver = self.r_proj(v_receiver)
        v_sender = self.s_proj(v_sender)
        x = x.transpose(0, 1)
        v_receiver = v_receiver.transpose(0, 1)
        v_sender = v_sender.transpose(0, 1)
        return x, v_receiver, v_sender