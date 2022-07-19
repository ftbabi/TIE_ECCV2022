model = dict(
    type='ParticleSimulator',
    backbone=dict(
        type='TIE',
        attr_dim=6,
        state_dim=6,
        position_dim=3,
        embed_dims=128,
        num_heads=8,
        num_encoder_layers=4,
        order=('selfattn', 'norm', 'ffn', 'norm')),
    head=dict(
        type='ParticleHead',
        # in_channels is nf_effect
        in_channels=128,
        # out_channels is position_dim
        out_channels=3,
        seperate=False,
        rotation_dim=0,
        weighted=False,
        loss=dict(type='MSELoss', loss_weight=1.0),),
)