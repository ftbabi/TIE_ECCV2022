dataset_type = 'PhysicsFleXDataset'

env_cfg = dict(
    # attn_mask = 8,
    env = 'FluidFall',
    num_abs_token=0,
    baseline=False,
    gen_data = False,
    gen_meta = False,
    hierarchy=False,
    scene_params = [],
    eval_ratio = 1e7,
    dataf = './data/FluidFall',
    n_rollout = 8,
    time_step = 121,
    time_step_clip = 5,
    attn_mask=8,

    dt = 1./60.,
    nf_relation = 300,
    nf_particle = 200,
    nf_effect = 200,
    train_valid_ratio = 0.5,

    n_instance = 1,
    n_stages = 1,
    n_his = 0,
    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    shape_state_dim = 14,
    # object attr:
    # [fluid]
    attr_dim = 3,
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    state_dim = 6,
    position_dim = 3,
    # relation attr:
    # [none]
    relation_dim = 1,
    
    neighbor_radius = 0.08,
    pstep = 2,
    phases_dict = dict(
        instance_idx = [0, 189],
        root_num = [[]],
        instance = ['fluid'],
        material = ['fluid']),
)



data = dict(
    samples_per_gpu=64,
    workers_per_gpu=10,
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