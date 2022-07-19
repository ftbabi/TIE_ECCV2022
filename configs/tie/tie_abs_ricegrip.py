_base_ = [
    './tie_ricegrip.py'
]

# Custom model
num_abs_token=2
model = dict(
    backbone=dict(
        num_abs_token=num_abs_token,),)

# Custom dataset
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    val=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    test=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),)