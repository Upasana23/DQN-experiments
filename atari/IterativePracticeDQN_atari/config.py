# transfer leanring configurations

CONFIG= {
    # Neural Network
    'filter1': 32,
    'filter2': 64,
    'filter3': 64,
    'units1': 512,
    'size1': 8,
    'size2':4,
    'size3': 3,
    'strides1': 4,
    'strides2': 2,
    'strides3': 1,
    'grad_clip': 10,
    
    #Practice
    'pre_iterations': 1000000,
    'pre_train_freq': 4,
    'pre_training_start': 50000,
    
    # Training
    'steps':400,
    'iterations': 200000,
    'update_target_freq': 10000,
    'train_freq': 4,
    'training_start': 50000,
    'eval_freq': 300000,
    'eval_episodes': 50,

    # Environment
    'env_id': 'BreakoutNoFrameskip-v4',
    #'env_id' : 'PongNoFrameskip-v4',
    #'env_id': 'FreewayNoFrameskip-v4',
    'discount_factor': 0.99,
    'frame_stack': 4,
    'noop_max': 30,

    # Replay Buffer
    'max_size': int(1e6),
    'batch_size': 64
}