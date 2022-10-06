dataset_type = 'PhysicsFleXDataset'

pstep = 2

env_cfg = dict(
    env = 'FluidShake',
    num_abs_token=0,
    hierarchy=False,
    eval_ratio = 1e7,
    dataf = './data/FluidShake/',
    n_rollout = 8,
    time_step = 301,
    time_step_clip = 0,
    attn_mask=8,

    dt = 1./60.,
    nf_relation = 300,
    nf_particle = 200,
    nf_effect = 200,
    train_valid_ratio = 0.5,

    n_instance = 1,
    n_stages = 2,
    n_his = 0,
    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    shape_state_dim = 14,
    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    attr_dim = 6,
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    state_dim = 6,
    position_dim = 3,
    # relation attr:
    # [none]
    relation_dim = 1,

    neighbor_radius = 0.08,
    pstep = pstep,
    phases_dict = dict(
        root_num = [[]],
        instance = ["fluid"],
        material = ["fluid"],),
)


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        phase='train',
        env_cfg=env_cfg,
        verbose=False,
    ),
    val=dict(
        type=dataset_type,
        phase='valid',
        env_cfg=env_cfg,
        verbose=False,
    ),
    test=dict(
        type=dataset_type,
        phase='valid',
        env_cfg=env_cfg,
        verbose=False,
    ),
)