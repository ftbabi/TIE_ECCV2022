dataset_type = 'PhysicsFleXDataset'

pstep = 2

env_cfg = dict(
    env = 'RiceGrip',
    num_abs_token=0,
    # If hierarchy = True, then is the original dataset
    hierarchy = False,
    gen_data = False,
    gen_meta = False,
    scene_params = [
        [8.0, 10.0], # x
        [8.0, 10.0], # y
        [8.0, 10.0], # z
    ],
    abs_point = False,
    eval_ratio = 1e7,
    dataf = './data/RiceGrip',
    n_rollout = 8,
    time_step = 41,
    time_step_clip = 0,
    attn_mask=8,

    dt = 1./60.,
    nf_relation = 300,
    nf_particle = 200,
    nf_effect = 200,
    train_valid_ratio = 0.5,

    n_instance = 1,
    n_stages = 4,
    n_his = 3,
    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    shape_state_dim = 14,
    # object attr:
    # [fluid, root, gripper_0, gripper_1,
    #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
    attr_dim = 7,
    # object state:
    # [rest_x, rest_y, rest_z, rest_xdot, rest_ydot, rest_zdot,
    #  x, y, z, xdot, ydot, zdot, quat.x, quat.y, quat.z, quat.w]
    # state_dim = 16 + 6 * args.n_his
    state_dim = 16 + 6 * 3,
    # rest_x, rest_y, rest_z, x, y, z
    position_dim = 6,
    # relation attr:
    # [none]
    relation_dim = 1,
    
    neighbor_radius = 0.08,
    pstep = pstep,
    phases_dict = dict(
        root_num = [[30]],
        root_sib_radius = [[5.0]],
        root_des_radius = [[0.2]],
        root_pstep = [[pstep]],
        instance = ['rice'],
        material = ['fluid'],),
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