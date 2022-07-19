_base_ = [
    '../_base_/models/tie.py',
    '../_base_/datasets/box_bath.py',
    '../_base_/schedules/adam_plateau_bs16.py',
    '../_base_/default_runtime.py'
]

find_unused_parameters = True

# Custom model
model = dict(
    backbone=dict(attr_dim=7,),
    head=dict(rotation_dim=4,),
)

# Custom dataset
data = dict(
    # Batch size = num_gpu * samples_per_gpu
    samples_per_gpu=2,
    workers_per_gpu=8,)

# Custom scheduler
runner = dict(max_epochs=5)
dist_params = dict(port='29511')

checkpoint_config = dict(interval=1)