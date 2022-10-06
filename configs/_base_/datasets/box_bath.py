dataset_type = 'PhysicsFleXDataset'
pstep = 2
env_cfg = dict(
    gen_data=False,
    scene_params = [0.25, 0.25, 0.25, 0.25, 0, 8, 15, 8],
    env = 'BoxBath',
    hierarchy=False,
    num_abs_token=0,
    eval_ratio = 1e7,
    dataf = './data/BoxBath',
    n_rollout = 8,
    time_step = 151,
    time_step_clip = 0,
    attn_mask=8,

    dt = 1./60.,
    nf_relation = 300,
    nf_particle = 200,
    nf_effect = 200,
    train_valid_ratio = 0.5,

    n_instance = 2,
    n_stages = 4,
    n_his = 0,
    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    shape_state_dim = 14,
    # object attr:
    # [rigid, fluid, wall * 5]
    attr_dim = 7,
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    state_dim = 6,
    position_dim = 3,
    # relation attr:
    # [none]
    relation_dim = 1,
    pstep=pstep,
    neighbor_radius = 0.08,
    phases_dict = dict(
        instance_idx = [0, 64, 1024],
        root_num = [[8], []],
        root_sib_radius = [[0.4], []],
        root_des_radius = [[0.08], []],
        root_pstep = [[pstep], []],
        instance = ['cube', 'fluid'],
        material = ['rigid', 'fluid'],),
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