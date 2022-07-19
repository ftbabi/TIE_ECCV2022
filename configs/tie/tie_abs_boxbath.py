_base_ = [
    './tie_boxbath.py',
]

# Custom model
num_abs_token = 2
model = dict(
    backbone=dict(
        num_abs_token=num_abs_token,),)

data = dict(
    train=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    val=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    test=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),)