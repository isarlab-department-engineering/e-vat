if __name__ == '__main__':

    from isarsocket.simple_socket import SimpleSocket
    from remind.remind import Remind
    from model_1CA import IMPALA_Net
    from learner import MyLearner
    from my_agent import MyAgent
    from gym import spaces

    import torch.multiprocessing as mp
    import numpy as np
    import random
    import torch
    import json
    import sys
    import os

    # load W&B data
    WANDB = False
    # switch between train and evaluation mode
    eval_mode = True

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Path for Models
    pathname = os.path.dirname(sys.argv[0])
    abs_path = os.path.abspath(pathname)

    algorithm = 'IMPALA'
    print(algorithm)
    net = IMPALA_Net

    json_paths = [['{}/settings/IMPALA/settings.json'.format(abs_path)],
                  ['{}/settings/IMPALA/settings.json'.format(abs_path), '{}/settings/IMPALA/settings_target.json'.format(abs_path)]]
    f = open('settings/IMPALA/settings.json')
    json_settings = json.load(f)

    print('MAIN', os.getpid())

    mp.set_start_method('spawn')

    # [sequence, batch, features]
    # hyperparams
    seq_len = 50
    memory_len = 1500
    episode_length = 20
    eval_episodes = 2
    N_Agents = 8
    n_actions = 8
    action_space = spaces.Discrete(8)
    batch_size = 24

    # socket
    main_to_agent_address = 6734
    agent_address_list = [('127.0.0.1', main_to_agent_address + i) for i in range(N_Agents + 2)]

    # trained model
    state_dict = 'two_tails_1CA.pt'

    # sensors settings
    RGB_width = json_settings['sensor_settings']['RGBCamera']['width']
    RGB_height = json_settings['sensor_settings']['RGBCamera']['height']
    grid_cells = json_settings['sensor_settings']['OccupancyGrid']['size_x']

    # settings dict
    settings = {
        'model_param': {
            'dl_1': 400,
            'dl_2': 300,
            'd1': 16,
            'd2': 32,
            'd3': 64,
            'dl': 256,
            'dr': 256,
            'channels': 3,
            'width': RGB_width,
            'height': RGB_height,
            'RGB_state_space': np.array([RGB_width, RGB_height, 3]),
            'state_space': 2,
            'init_w': 3e-3,
            'output_a': n_actions,
            'exp_distance': 50,
            'max_distance': 20,
            'TD3': False,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'grid_cells': grid_cells,
            'resnet': False,
        },
        'agent_gpu_id': 0,
        'update_steps': 100,
        'env_track': 'unrealisar-track-v0',
        'env_exp': 'unrealisar-exp-v0',
        'env_name': 'unrealisar-lg-v0',
        'multi_mod': True,
        'unreal': {
            'start_port_test': 9734,
            'start_port_train': 9736,
            'json_paths': json_paths,
            'render': True,
            'address': 'localhost'
        },
        'test': True,
        'fov_gt_test': False,
        'hysteresis': False,
        'eval_mode': eval_mode,
        'eval_episodes': eval_episodes,
        'seq_len': seq_len,
        'memory_len': memory_len,
        'action_space': action_space,
        'action_space_noise': spaces.Box(low=-1, high=1, shape=(1,)),
        'WandB': WANDB,
        'workstation_name': 'Perseo',
        'main_to_agent_address': ('127.0.0.1', main_to_agent_address),
        'first_address': ('127.0.0.1', 7734),
        'learner_address': ('127.0.0.1', 8734),
        'episode_length': episode_length,
        'warmup': seq_len * batch_size + batch_size - 1 + int((seq_len + batch_size - 1) / episode_length) * (seq_len - 1),
        'on_policy': False,
        'learner_rank': 0,
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'lr_impala': 0.0002,
        'batch_size': batch_size,
        'learner_gpu_id': 0,
        'memory_address': ('127.0.0.1', 7734 + N_Agents),
        'num_agents': N_Agents,
        'tau': 0.01,
        'gamma': 0.99,
        'RMSprop_eps': 0.1,
        'algorithm': algorithm,
        'baseline_cost': 0.5,
        'entropy_cost': 0.001,
        'navigation_cost': 1,
        'obstacle_rec_cost': 1,
        'tracker_rec_cost': 1e2,
        'target_rec_cost': 1e2,
        'TD3': False,
        'IMPALA': True,
        'update_settings': {
            'batch_size': batch_size,
            'min_new_samples': 100 * N_Agents,
            'mode': 'newest',
            'data_keys': {
                'RGBCamera': seq_len,
                'action_track': seq_len,
                'action_exp': seq_len,
                'reward_track': seq_len,
                'reward_exp': seq_len,
                'next_state': 1,
                'done_track': seq_len,
                'done_exp': seq_len,
                'track_h_a': 1,
                'exp_h_a': 1,
                'fov_h_a': 1,
                'track_h_c': 1,
                'exp_h_c': 1,
                'logits_track': seq_len,
                'logits_exp': seq_len,
                'yaw': seq_len,
                'ego_target': seq_len,
                'angle': seq_len,
                'distance': seq_len,
                'hit': seq_len,
                'fov_gt': seq_len,
                'bel': seq_len,
                'track_flag': seq_len,
            }
        },
        'agents_fps_array': mp.Array('d', np.zeros(N_Agents)),
        'agent_rank': mp.Value('i', 0),
        'shared_counter': mp.Value('i', 0),
        'grid_cells': grid_cells,
        'obstacle_ch_weight': 1,
        'tracker_ch_weight': 1,
        'target_ch_weight': 2,
        'state_dict': state_dict
    }

    pids = []

    test_agent = MyAgent(settings)
    test_agent.start()
    pids.append(test_agent.pid)

    settings['test'] = False

    if not eval_mode:
        memory = Remind(first_address=('127.0.0.1', 7734), num_agents=N_Agents, seed=10)
        memory.start()
        pids.append(memory.pid)

        agents = [MyAgent(settings=settings) for i in range(N_Agents)]
        learner = MyLearner(settings=settings)

        [agent.start() for agent in agents]

        learner.start()
        pids.append(learner.pid)
        [pids.append(agent.pid) for agent in agents]

        in_socket = SimpleSocket(address_list=agent_address_list, server=True, name='main_to_agents')

        # Net
        gnet = net(settings=settings['model_param'])

        # test
        in_socket.server_send({'net': gnet}, socket_id=0)

        # train gnet
        for i in range(N_Agents):
            in_socket.server_send({'net': gnet}, socket_id=i + 1)

        # learner
        in_socket.server_send({'net': gnet}, socket_id=N_Agents + 1)

        del gnet

        [agent.join() for agent in agents]
        learner.join()

    else:
        gnet = net(settings=settings['model_param'])
        gnet.load_state_dict(torch.load('models/' + state_dict, map_location='cuda:0'))

        in_socket = SimpleSocket(address_list=[('127.0.0.1', main_to_agent_address)],
                                 server=True,
                                 name='main_to_agents')
        in_socket.server_send({'net': gnet}, socket_id=0)
        del gnet

    test_agent.join()

