import math
import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer)

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from sim.models.utils import (FFN, AttentionTIE, kaiming_uniform_, kaiming_normal_)

# from .builder import TRANSFORMER
class MultiheadAttention(nn.Module):
    """
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default 0.0.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0, **kwargs):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = AttentionTIE(embed_dims, num_heads, dropout, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                output_mask=None,
                receiver_val_res=None, sender_val_res=None,
                residual_receiver=None, residual_sender=None
                , **kwargs):
        """Forward function for `MultiheadAttention`.
        Args:
            x (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. Default None. If not None, it will
                be added to `x` before forward function.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Default None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (Tensor): ByteTensor mask with shape [num_query,
                num_key]. Same in `nn.MultiheadAttention.forward`.
                Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `nn.MultiheadAttention.forward`. Default None.
            receiver_val_res: n_particles, bs, embed_dims
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        out, v_receiver, v_sender = self.attn(query, attn_mask, key_padding_mask, output_mask, receiver_val_res=receiver_val_res, sender_val_res=sender_val_res,
            residual_receiver=residual_receiver, residual_sender=residual_sender, **kwargs)

        return residual + self.dropout(out), v_receiver, v_sender


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in transformer.
    Args:
        embed_dims (int): The feature dimension. Same as `FFN`.
        num_heads (int): Parallel attention heads.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
        order (tuple[str]): The order for encoder layer. Valid examples are
            ('selfattn', 'norm', 'ffn', 'norm') and ('norm', 'selfattn',
            'norm', 'ffn'). Default ('selfattn', 'norm', 'ffn', 'norm').
        act_cfg (dict): The activation config for FFNs. Default ReLU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = order[0] == 'norm'
        self.self_attn = MultiheadAttention(embed_dims, num_heads, dropout, act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)
        self.ffn = FFN([embed_dims, 2 * embed_dims, embed_dims], final_act=True, bias=True, add_residual=True)
        self.norms = nn.ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.res_norms = nn.ModuleList()
        self.res_norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        self.res_norms.append(build_norm_layer(norm_cfg, embed_dims)[1])

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None, output_mask=None, receiver_val_res=None, sender_val_res=None, **kwargs):
        """Forward function for `TransformerEncoderLayer`.
        Args:
            x (Tensor): The input query with shape [num_key, bs,
                embed_dims]. Same in `MultiheadAttention.forward`.
            pos (Tensor): The positional encoding for query. Default None.
                Same as `query_pos` in `MultiheadAttention.forward`.
            attn_mask (Tensor): ByteTensor mask with shape [num_key,
                num_key]. Same in `MultiheadAttention.forward`. Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `MultiheadAttention.forward`. Default None.
            receiver_val_res: n_particles, bs, embed_dims
        """
        norm_cnt = 0
        inp_residual = x
        inp_res_r = receiver_val_res
        inp_res_s = sender_val_res
        if self.pre_norm:
            x = self.norms[norm_cnt](x)
            norm_cnt += 1
            if receiver_val_res is not None and sender_val_res is not None:
                receiver_val_res = self.res_norms[0](receiver_val_res)
                sender_val_res = self.res_norms[1](sender_val_res)

        # self attn
        query = key = value = x
        x, receiver_val_res, sender_val_res = self.self_attn(
            query,
            key,
            value,
            inp_residual if self.pre_norm else None,
            query_pos=pos,
            key_pos=pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            output_mask=output_mask,
            receiver_val_res=receiver_val_res, sender_val_res=sender_val_res,
            residual_receiver=inp_res_r, residual_sender=inp_res_s, **kwargs)
        inp_residual = x

        # norm
        x = self.norms[norm_cnt](x)
        norm_cnt += 1

        # ffn
        x = self.ffn(x, inp_residual if self.pre_norm else None)

        # norm
        if not self.pre_norm:
            x = self.norms[norm_cnt](x)
            norm_cnt += 1
            if receiver_val_res is not None and sender_val_res is not None:
                receiver_val_res = self.res_norms[0](receiver_val_res)
                sender_val_res = self.res_norms[1](sender_val_res)

        return x, receiver_val_res, sender_val_res


class TransformerEncoder(nn.Module):
    """Implements the encoder in transformer.
    Args:
        num_layers (int): The number of `TransformerEncoderLayer`.
        embed_dims (int): Same as `TransformerEncoderLayer`.
        num_heads (int): Same as `TransformerEncoderLayer`.
        dropout (float): Same as `TransformerEncoderLayer`. Default 0.0.
        order (tuple[str]): Same as `TransformerEncoderLayer`.
        act_cfg (dict): Same as `TransformerEncoderLayer`. Default ReLU.
        norm_cfg (dict): Same as `TransformerEncoderLayer`. Default
            layer normalization.
    """

    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'), **kwargs):
        super(TransformerEncoder, self).__init__()
        assert isinstance(order, tuple) and len(order) == 4
        assert set(order) == set(['selfattn', 'norm', 'ffn'])
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.order = order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = order[0] == 'norm'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims, num_heads, dropout, order, act_cfg, norm_cfg, **kwargs))
        self.norm = build_norm_layer(norm_cfg,
                                     embed_dims)[1] if self.pre_norm else None

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None, output_mask=None, receiver_val_res=None, sender_val_res=None, **kwargs):
        """Forward function for `TransformerEncoder`.
        Args:
            x (Tensor): Input query. Same in `TransformerEncoderLayer.forward`.
            pos (Tensor): Positional encoding for query. Default None.
                Same in `TransformerEncoderLayer.forward`.
            attn_mask (Tensor): ByteTensor attention mask. Default None.
                Same in `TransformerEncoderLayer.forward`.
            key_padding_mask (Tensor): Same in
                `TransformerEncoderLayer.forward`. Default None.
            receiver_val_res: n_particles, bs, embed_dims
        """
        for layer in self.layers:
            x, receiver_val_res, sender_val_res = layer(x, pos, attn_mask, key_padding_mask, output_mask, receiver_val_res, sender_val_res, **kwargs)
        if self.norm is not None:
            x = self.norm(x)
        return x


@BACKBONES.register_module()
class TIE(BaseBackbone):
    """Implements the simulation transformer.
    """

    def __init__(self,
                 attr_dim,
                 state_dim,
                 position_dim,
                 embed_dims, 
                 num_heads=8,
                 num_encoder_layers=4,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 norm_eval=False, num_abs_token=0, **kwargs):
        super(TIE, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.num_abs_token = num_abs_token

        self.norm_eval = norm_eval
        self.input_projection = FFN(
            [attr_dim + state_dim, embed_dims], 
            final_act=True, bias=True)

        fan_in = (attr_dim + state_dim)
        fan_in = 2 * (attr_dim + state_dim)
        self.rs_weight = nn.Parameter(torch.Tensor(embed_dims, 2 * attr_dim + 2 * state_dim))
        nn.init.kaiming_uniform_(self.rs_weight, a=math.sqrt(5))

        if norm_cfg is not None:
            self.receiver_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.sender_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.particle_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.receiver_norm = nn.Identity()
            self.sender_norm = nn.Identity()
            self.particle_norm = nn.Identity()
        
        if self.num_abs_token > 0:
            assert self.num_abs_token == 2
            self.abs_token = nn.Parameter(torch.Tensor(self.num_abs_token, attr_dim + state_dim))
            nn.init.zeros_(self.abs_token)

        self.encoder = TransformerEncoder(num_encoder_layers, embed_dims,
                                          num_heads,
                                          dropout, order, act_cfg,
                                          norm_cfg, **kwargs)

    def forward(self, attr, state, fluid_mask, rigid_mask, output_mask, attn_mask, **kwargs):
        """Forward function for `TIE`.
        """
        x = torch.cat([attr.squeeze(1).transpose(-1, -2), state.squeeze(1).transpose(-1, -2)], dim=-1)
        if self.num_abs_token > 0:
            x[:, -self.num_abs_token:] = self.abs_token
            abs_mask = torch.cat([rigid_mask, fluid_mask], dim=1).unsqueeze(1)
            # pad_mask[:, :, -self.num_abs_token:] = ~(abs_mask.sum(dim=-1) > 0)
            abs_mask = ~(abs_mask.bool())
            attn_mask[:, :, -self.num_abs_token:] = abs_mask
            attn_mask[:, :, :, -self.num_abs_token:] = abs_mask.transpose(-1, -2)

        r_r_w = self.rs_weight[:, :self.attr_dim + self.state_dim]
        r_s_w = self.rs_weight[:, self.attr_dim + self.state_dim:]
        receiver_val_res = x.matmul(r_r_w.t())
        sender_val_res = x.matmul(r_s_w.t())
        receiver_val_res = receiver_val_res.permute(1, 0, 2)
        sender_val_res = sender_val_res.permute(1, 0, 2)

        x = self.input_projection(x)

        x = x.permute(1, 0, 2)  # [bs, n_p, c] -> [n_p, bs, c]

        x_enc = self.encoder(
            x, 
            attn_mask=attn_mask, key_padding_mask=None, output_mask=output_mask, 
            receiver_val_res=receiver_val_res, sender_val_res=sender_val_res)
        x_enc = x_enc.permute(1, 0, 2)
        return x_enc

    def train(self, mode=True):
        super(TIE, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()