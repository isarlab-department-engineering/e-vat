from utils import create_env, UniformNoise, ScoreCounter, Belief
from isarsocket.simple_socket import SimpleSocket

import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import wandb
import random
import torch
import copy
import time
import sys
import os


def get_sockets(rank, first_address, learner_address, main_address, eval_mode):
    if rank != 0:
        remind_address = [(first_address[0], first_address[1] + rank - 1)]
    learner_in_address = [(learner_address[0], learner_address[1] + rank)]
    learner_out_address = [(learner_address[0], learner_address[1] + 100 + rank)]
    main_address = [(main_address[0], main_address[1] + rank)]
    main_socket = SimpleSocket(address_list=main_address, server=False, name='main')
    isar_socket_memory = SimpleSocket(address_list=remind_address, server=False,
                                      name='agent_mem') if rank != 0 else None
    learner_in_socket = SimpleSocket(address_list=learner_in_address, server=False,
                                     name='agent_in_learner') if not eval_mode else None
    learner_out_socket = SimpleSocket(address_list=learner_out_address, server=False,
                                      name='agent_out_learner') if not eval_mode else None

    return isar_socket_memory, learner_in_socket, learner_out_socket, main_socket


def get_plot_sockets(plot_address, plot_list):
    grid_plot_in_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1])], server=True,
                                       name='grid_socket') if plot_list[0] else None
    reward_plot_in_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1] + 1)], server=True,
                                         name='reward_socket') if plot_list[1] else None
    maprec_plot_in_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1] + 2)], server=True,
                                         name='graph_socket') if plot_list[2] else None

    return grid_plot_in_socket, reward_plot_in_socket, maprec_plot_in_socket


class MyAgent(mp.Process):
    def __init__(self, settings):
        super(MyAgent, self).__init__()
        self.daemon = True

        self.settings = settings

        with self.settings['agent_rank'].get_lock():
            self.rank = self.settings['agent_rank'].value
            self.settings['agent_rank'].value += 1

        self.noise = UniformNoise(self.settings['action_space_noise'], max_sigma=0.5, min_sigma=0.5)
        self.epsilon, self.depsilon = 1, 1.0 / 50000

        self.model_update = 0
        self.update_data = {}

        self.main_socket = None
        self.learner_in_socket = None
        self.learner_out_socket = None
        self.isar_socket_memory = None
        self.plot_grid_in_socket = None
        self.plot_rew_in_socket = None
        self.plot_maprec_in_socket = None

        self.env = None

        self.n_step = 0

        # Path for Models
        pathname = os.path.dirname(sys.argv[0])
        self.abs_path = os.path.abspath(pathname)

        # for multi-GPUs machines
        # if self.rank >= 4:
        #     self.settings['agent_gpu_id'] = 1 - self.settings['agent_gpu_id']

        if torch.cuda.is_available() and self.settings['agent_gpu_id'] is not None:
            self.device = torch.device('cuda:' + str(self.settings['agent_gpu_id']))
        else:
            self.device = torch.device('cpu')

        print(self.settings['agent_gpu_id'], self.rank)

        self.belief = Belief(self.settings, self.device)
        self.models = None
        self.model = None
        self.reward_ep, self.episodes = 0, 0
        self.start_time = time.time()
        if self.settings['eval_mode'] and self.settings['test']:
            self.score_counter = ScoreCounter(dis_exp=50, dis_max=90)

    def update_model(self):
        if self.settings['shared_counter'].value == self.settings['warmup'] and self.n_step >= (
                self.settings['warmup'] if self.settings['on_policy'] else self.settings['warmup'] / self.settings[
                    'num_agents']):
            # off policy
            if not self.settings['on_policy']:
                self.learner_in_socket.client_send(
                    ['send_last_state_dict', {'model_update': self.model_update, 'socket_id': self.rank}])
            # on policy
            else:
                if self.model.learnable:
                    self.learner_in_socket.client_send(['on_policy_update',
                                                        {'agent_state_dict': self.model.state_dict(),
                                                         'memory_id': self.rank, 'socket_id': self.rank}])
                else:
                    self.learner_in_socket.client_send(
                        ['off_policy_update', {'memory_id': self.rank, 'socket_id': self.rank}])
            # learner new_data
            update_data = self.learner_out_socket.client_receive()
            # off policy
            if not self.settings['on_policy']:
                self.model_update = update_data['model_update']
            # load state_dict
            self.model.load_state_dict(update_data['last_state_dict_te']) if self.model.learnable else None
            # update data
            for k in update_data.get('log', {}):
                self.update_data['train/{}'.format(k)] = self.update_data.get('train/{}'.format(k), 0) + \
                                                         update_data['log'][k]

    def upgrade_model_from_shared(self):
        if self.settings['shared_counter'].value == self.settings['warmup'] and self.n_step >= self.settings['warmup']:
            self.learner_in_socket.client_send(
                ['send_last_state_dict', {'model_update': self.model_update, 'socket_id': self.rank}])
            update_data = self.learner_out_socket.client_receive()

            self.model_update = update_data['model_update']
            self.model.load_state_dict(update_data['last_state_dict_te'])

    def save_model(self, save_id):
        torch.save(self.model.state_dict(), './models/best_model_{}.pt'.format(save_id))

    def run(self):
        torch.manual_seed(self.rank)
        torch.cuda.manual_seed(self.rank)
        np.random.seed(self.rank)
        random.seed(self.rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.isar_socket_memory, self.learner_in_socket, self.learner_out_socket, self.main_socket = \
            get_sockets(self.rank, self.settings['first_address'], self.settings['learner_address'],
                        self.settings['main_to_agent_address'], self.settings['eval_mode'])

        if not self.settings['test']:
            memory_settings = {
                'memory_id': self.rank,
                'seed': self.rank,
                'max_size': self.settings['memory_len'],
                'key_list': [
                    'RGBCamera',
                    'next_state',
                    'action_track',
                    'action_exp',
                    'reward_track',
                    'reward_exp',
                    'done_track',
                    'done_exp',
                    'track_h_a',
                    'exp_h_a',
                    'fov_h_a',
                    'track_h_c',
                    'exp_h_c',
                    'logits_track',
                    'logits_exp',
                    'yaw',
                    'ego_target',
                    'angle',
                    'distance',
                    'hit',
                    'fov_gt',
                    'bel',
                    'track_flag',
                ],
                'sequence_length': self.settings['seq_len']
            }
            self.isar_socket_memory.client_send(['init_memory', memory_settings])

        print('test' if self.settings['test'] else 'train', os.getpid())

        reward_ep, reward_exp, reward_track, episodes = 0, 0, 0, 0
        last_reward = np.array([0])

        self.models = self.main_socket.client_receive()
        self.model = self.models['net'].to(self.device)

        tracking_agent = self.rank % 2 == 0

        if self.settings['multi_mod']:
            if tracking_agent:
                self.settings['unreal']['ports'] = [(self.settings['unreal']['start_port_test'] + self.rank * 2),
                                                    (self.settings['unreal']['start_port_test'] + self.rank * 2) + 1]
                self.settings['unreal']['json_paths'] = self.settings['unreal']['json_paths'][1]
                self.settings['env_name'] = self.settings['env_track']
                if not self.settings['test']:
                    self.settings['episode_length'] = 900
            else:
                self.settings['unreal']['ports'] = [(self.settings['unreal']['start_port_test'] + self.rank * 2)]
                self.settings['unreal']['json_paths'] = self.settings['unreal']['json_paths'][0]
                self.settings['env_name'] = self.settings['env_exp']
        else:
            self.settings['unreal']['ports'] = [(self.settings['unreal']['start_port_test'] + self.rank * 2)]
            self.settings['unreal']['json_paths'] = self.settings['unreal']['json_paths'][0]

        print('environment', self.settings['unreal']['ports'])

        self.env = create_env(self.settings, seed=self.rank)
        self.env.action_space = self.settings['action_space']

        # reset
        state_info = self.env.reset()
        state, info = state_info['states'], state_info['info']

        # #############-init-###############
        # gru
        track_h_a = self.model.init_hidden().to(self.device)
        exp_h_a = self.model.init_hidden().to(self.device)
        fov_h_a = self.model.init_hidden().to(self.device)
        track_h_c = self.model.init_hidden().to(self.device)
        exp_h_c = self.model.init_hidden().to(self.device)
        # ground truth
        obstacle_gt = torch.from_numpy(info['Geo_target'][0]).to(self.device)
        tracker_gt = torch.from_numpy(info['Geo_target'][1]).to(self.device)  # tmp
        ego_target = torch.from_numpy(info['Ego_target']).to(self.device)  # [3, 11, 21]
        yaw = torch.from_numpy(info['GPS_Yaw']).float().to(self.device)
        angle = torch.from_numpy(info['angle']).float().to(self.device)
        distance = torch.from_numpy(info['distance']).float().to(self.device)
        hit = torch.from_numpy(info['hit']).float().to(self.device)
        fov_gt = torch.Tensor([0]).to(self.device)
        # belief
        ego_target_bel = torch.from_numpy(info['Ego_target']).to(self.device)
        yaw_bel = torch.from_numpy(info['GPS_Yaw']).float().to(self.device)
        position_bel = torch.zeros(1, 2).to(self.device)
        belief = self.belief.init_target_prob(torch.from_numpy(info['Geo_target']).to(self.device))
        ####################################

        self.settings['episode_length'] = 70 if not self.model.learnable else self.settings['episode_length']

        episode_time = time.time()
        step_in_episode = 0
        best_scores = []
        coverage = np.empty(0)

        # plot in eval_mode
        if self.settings['test']:
            if self.settings['eval_mode']:
                cv.namedWindow('geo_target', cv.WINDOW_NORMAL)
                cv.resizeWindow('geo_target', 300, 300)
                cv.namedWindow('ego_target', cv.WINDOW_NORMAL)
                cv.resizeWindow('ego_target', 300, 150)  # default_dimension
                cv.namedWindow('belief', cv.WINDOW_NORMAL)
                cv.resizeWindow('belief', 300, 300)
                cv.namedWindow('RGB', cv.WINDOW_NORMAL)
                cv.resizeWindow('RGB', 250, 250)

        hyst_threshold = 20
        exp_hyst = True
        exp_counter = hyst_threshold
        times = []

        while True:
            plt.pause(0.0001)

            if step_in_episode == 1:
                belief = self.belief.init_target_prob(torch.from_numpy(info['Geo_target']).to(self.device))

            self.n_step += 1
            step_in_episode += 1
            if self.settings['shared_counter'].value < self.settings['warmup'] and not self.settings['test']:
                with self.settings['shared_counter'].get_lock():
                    self.settings['shared_counter'].value += 1

            belief_up = torch.cat((obstacle_gt.unsqueeze(0), tracker_gt.unsqueeze(0), belief[2].unsqueeze(0)),
                                  dim=0).to(self.device)

            if torch.abs(angle) < 40 and distance < 130 and hit:
                fov_gt = torch.Tensor([1]).to(self.device)

            # model input
            input_data = {'device': self.device, 'agent_gpu_id': self.settings['agent_gpu_id'],
                          'RGBCamera': state,
                          'track_h_a': track_h_a.detach().cpu().numpy(),
                          'exp_h_a': exp_h_a.detach().cpu().numpy(),
                          'fov_h_a': fov_h_a.detach().cpu().numpy(),
                          'track_h_c': track_h_c.detach().cpu().numpy(),
                          'exp_h_c': exp_h_c.detach().cpu().numpy(),
                          'ego_target': ego_target,
                          'yaw': yaw / 180.,
                          'angle': angle / 180.,
                          'distance': distance / 50.,
                          'hit': hit,
                          'bel': belief_up,
                          }

            # model output
            a = time.time()
            output_data = self.model.choose_action(input_data)
            b = time.time()
            times.append(b-a)

            # update bel
            belief = self.belief.update_target_prob(belief, ego_target_bel, position_bel, yaw_bel,
                                                    torch.from_numpy(info['GeoEgo_Target']).to(self.device), gt=True)

            # target map
            geo_target = torch.from_numpy(info['Geo_target']).to(self.device)  # [3, grid_cells, grid_cells]

            # selection between tracking and exploration actions
            if self.settings['multi_mod']:
                if self.settings['test']:
                    if self.settings['fov_gt_test']:
                        if fov_gt > 0:
                            action_tracker = output_data['action_track']
                            if self.settings['eval_mode']:
                                print('fov_gt action_track: ', action_tracker)
                        else:
                            action_tracker = output_data['action_exp']
                            if self.settings['eval_mode']:
                                print('fov_gt action_exp: ', action_tracker)
                    else:
                        if torch.sigmoid(output_data['fov']) > 0.8:
                            action_tracker = output_data['action_track']
                            # if self.settings['eval_mode']:
                            #     print('fov_hat action_track: ', action_tracker)
                            exp_counter = 0
                        else:
                            if self.settings['hysteresis']:
                                if exp_hyst:
                                    action_tracker = output_data['action_exp']
                                    # if self.settings['eval_mode']:
                                    #     print('fov_hat action_exp: ', action_tracker)
                                else:
                                    action_tracker = output_data['action_track']
                                    # if self.settings['eval_mode']:
                                    #     print('fov_hat action_track: ', action_tracker)
                                exp_counter += 1
                            else:
                                action_tracker = output_data['action_exp']
                                # if self.settings['eval_mode']:
                                #     print('fov_hat action_exp: ', action_tracker)
                else:
                    if tracking_agent:
                        action_tracker = output_data['action_track']
                    else:
                        action_tracker = output_data['action_exp']
            else:
                action_tracker = output_data['action']

            if exp_counter >= hyst_threshold:
                exp_hyst = True
            else:
                exp_hyst = False

            if not self.settings['test'] \
                    and self.settings['algorithm'] not in ['A3C'] and self.settings['algorithm'] not in ['IMPALA']:
                action_tracker = [self.noise.get_action(action_tracker[0], t=self.n_step),
                                  self.noise.get_action(action_tracker[1], t=self.n_step)]
                self.epsilon -= self.depsilon

            # gym step
            new_state, done, new_info, reward = self.env.step(action_tracker)

            self.score_counter.add_score(info=new_info) if self.settings['eval_mode'] and self.settings['test'] else {}

            bel_max = torch.max(belief[2]).cpu().numpy()
            reward_bel = np.array([np.clip(bel_max, a_min=0, a_max=0.01) * 80000 / geo_target[2].sum().cpu().numpy()])

            if reward_bel > last_reward:
                last_reward = reward_bel
            else:
                reward_bel = np.array([0])
            bel_done = (bel_max >= 0.20 and step_in_episode > 51)

            done = 1 if step_in_episode == self.settings['episode_length'] or reward_ep < -450 or \
                        (step_in_episode > 51 and done and not self.settings['test']) or \
                        (bel_done and not tracking_agent and not self.settings['eval_mode']) else 0

            if bel_done and not tracking_agent:
                reward_bel = np.array([50])

            if self.settings['multi_mod']:
                if self.settings['test']:
                    reward_exp += reward_bel
                    reward = reward + reward_bel
                if tracking_agent:
                    reward_track += reward
                else:
                    reward += reward_bel
                    reward_exp += reward

            reward_ep += reward

            reward = reward.cpu().numpy() if type(reward) != np.ndarray else reward
            assert reward.shape == (1,)

            # remind
            data = {
                'memory_id': self.rank,
                'data': {
                    'RGBCamera': state,
                    'next_state': new_state,
                    'action_track': output_data['action_track'],
                    'action_exp': output_data['action_exp'],
                    'reward_track': reward,
                    'reward_exp': reward,
                    'done_track': done,
                    'done_exp': done,
                    'track_h_a': track_h_a.cpu().numpy(),
                    'exp_h_a': exp_h_a.cpu().numpy(),
                    'fov_h_a': fov_h_a.cpu().numpy(),
                    'track_h_c': track_h_c.cpu().numpy(),
                    'exp_h_c': exp_h_c.cpu().numpy(),
                    'logits_track': output_data['logits_track'],
                    'logits_exp': output_data['logits_exp'],
                    'yaw': yaw.cpu().numpy() / 180.,
                    'angle': angle.cpu().numpy() / 180.,
                    'distance': distance.cpu().numpy() / 50.,
                    'hit': hit.cpu().numpy(),
                    'ego_target': ego_target.cpu().numpy(),
                    'bel': belief_up.cpu().numpy(),
                    'fov_gt': fov_gt.cpu().numpy(),
                    'track_flag': tracking_agent,
                }
            }

            if not self.settings['test']:
                self.isar_socket_memory.client_send(['insert', data])
                if self.n_step % self.settings['update_steps'] == 0:
                    self.update_model()

            info_ego_target = torch.from_numpy(new_info['Ego_target']).to(self.device)

            if done:
                episodes += 1
                curr_time = time.time()
                interval = curr_time - episode_time

                exp_counter = hyst_threshold
                exp_hyst = True

                geo_grid = [wandb.Image(geo_target, caption='Geo_grid episode: {}'.format(episodes))]
                ego_grid = [wandb.Image(info_ego_target, caption='Ego_target episode: {}'.format(episodes))]
                bel_grid = [wandb.Image(belief[2] * 255, caption='Belief episode: {}'.format(episodes))]

                if self.settings['WandB'] and not self.settings['eval_mode']:
                    if self.settings['test']:
                        training_log = {
                            'test/reward_ep': reward_ep,
                            'test/reward_exp': reward_exp,
                            'test/reward_track': reward_track,
                            'test/fps': step_in_episode / interval,
                            'geo_target': geo_grid,
                            'ego_target': ego_grid,
                            'belief': bel_grid
                        }
                    else:
                        self.settings['agents_fps_array'][self.rank - 1] = step_in_episode / interval
                        if tracking_agent:
                            training_log = {
                                **{'train/reward_ep': reward_ep,
                                   'train/reward_track': reward_track,
                                   'train/fps': np.sum(self.settings['agents_fps_array'])},
                                # **{k.replace('train/', ''): v / step_in_episode for k, v in self.update_data.items()}
                                **{k: v / step_in_episode for k, v in self.update_data.items()}
                            }
                        else:
                            training_log = {
                                **{'train/reward_ep': reward_ep,
                                   'train/reward_exp': reward_exp,
                                   'train/fps': np.sum(self.settings['agents_fps_array'])},
                                **{k.replace('train/', ''): v / step_in_episode for k, v in self.update_data.items()}
                                # **{k: v / step_in_episode for k, v in self.update_data.items()}
                            }
                    self.learner_in_socket.client_send(['save_log', {'training_log': training_log}])

                self.noise.reset()

                if self.settings['eval_mode'] and self.settings['test']:
                    self.score_counter.reset()
                    if episodes == self.settings['eval_episodes']:
                        eval_time = time.time() - self.start_time
                        self.score_counter.save_scores(self.settings['eval_episodes'], self.settings['episode_length'],
                                                       eval_time, self.settings['state_dict'])

                step_in_episode = 0

                new_state_info = self.env.reset()
                new_state = new_state_info['states']
                new_info = new_state_info['info']

                self.update_data = {}

                if self.settings['test']:
                    if not self.settings['eval_mode']:
                        if len(best_scores) < 10:
                            self.save_model(len(best_scores))
                            print("New model saved")
                            best_scores.append(reward_ep)
                        elif reward_ep > min(best_scores):
                            save_id = best_scores.index(min(best_scores))
                            self.save_model(save_id)
                            print("New model saved")
                            best_scores[save_id] = reward_ep
                        self.upgrade_model_from_shared()

                output_data['track_h_a_'] = self.model.init_hidden()
                output_data['exp_h_a_'] = self.model.init_hidden()
                output_data['fov_h_a_'] = self.model.init_hidden()
                output_data['track_h_c_'] = self.model.init_hidden()
                output_data['exp_h_c_'] = self.model.init_hidden()

                belief = self.belief.init_target_prob(geo_target)

                if self.settings['test']:
                    coverage = np.append(coverage, reward_ep)
                    # print('\n{} episodes'.format(episodes))
                    # print('average: {}, max: {}, min: {}'.format(int(coverage.mean()), int(coverage.max()),
                    #                                              int(coverage.min())))

                reward_ep, reward_exp, reward_track = 0, 0, 0
                last_reward = np.array([0])
                episode_time = time.time()

            state = new_state
            info = new_info

            if self.settings['test']:
                if self.settings['eval_mode']:
                    # global map
                    gt = torch.clamp(geo_target, min=0, max=255).permute(1, 2, 0).detach().cpu().numpy()
                    img = cv.cvtColor(gt, cv.COLOR_BGR2RGB)
                    cv.imshow('geo_target', img)
                    # first-person map
                    et = torch.clamp(info_ego_target, min=0, max=255).permute(1, 2, 0).detach().cpu().numpy()
                    img = cv.cvtColor(et, cv.COLOR_BGR2RGB)
                    cv.imshow('ego_target', img)
                    # belief map
                    bel = torch.clamp(belief[2] * 255, min=0., max=255.).unsqueeze(-1).detach().cpu().numpy()
                    img = cv.cvtColor(bel, cv.COLOR_BGR2RGB)
                    cv.imshow('belief', img)
                    # first-person camera
                    text = 'fov ' + str(torch.sigmoid(output_data['fov'][0][0]).detach().cpu().numpy())
                    text_pos = (3, 10)
                    text_font = cv.FONT_ITALIC
                    text_fontscale = 0.3
                    if torch.sigmoid(output_data['fov'][0][0]) < 0.8:
                        text_color = (0, 0, 255)  # BGR
                    else:
                        text_color = (0, 255, 0)  # BGR
                    text_thickness = 1
                    st = torch.clamp(torch.from_numpy(state), min=0., max=255.).cpu().numpy()
                    img = cv.cvtColor(st, cv.COLOR_BGR2RGB)
                    cv.putText(img, text, text_pos, text_font, text_fontscale, text_color, text_thickness, cv.LINE_AA)
                    cv.imshow('RGB', img)
                    #
                    cv.waitKey(1)

            # ##################-data update-######################
            track_h_a = copy.deepcopy(output_data['track_h_a_'].detach().cpu())
            exp_h_a = copy.deepcopy(output_data['exp_h_a_'].detach().cpu())
            fov_h_a = copy.deepcopy(output_data['fov_h_a_'].detach().cpu())
            track_h_c = copy.deepcopy(output_data['track_h_c_'].detach().cpu())
            exp_h_c = copy.deepcopy(output_data['exp_h_c_'].detach().cpu())
            obstacle_gt = torch.from_numpy(info['Geo_target'][0]).to(self.device)
            tracker_gt = torch.from_numpy(info['Geo_target'][1]).to(self.device)
            yaw = torch.from_numpy(info['GPS_Yaw']).float().to(self.device)
            angle = torch.from_numpy(info['angle']).float().to(self.device)
            distance = torch.from_numpy(info['distance']).float().to(self.device)
            hit = torch.from_numpy(info['hit']).float().to(self.device)
            fov_gt = torch.Tensor([0]).to(self.device)
            ego_target = torch.from_numpy(info['Ego_target']).to(self.device)
            ego_target_bel = torch.from_numpy(info['Ego_target']).to(self.device)
            yaw_bel = torch.from_numpy(info['GPS_Yaw']).float().to(self.device)
            position_bel = torch.from_numpy(info['Tracker_position']).to(self.device)
            #######################################################

        self.env.close()
        print('Agent %i finished after %i steps.' % (self.rank, self.n_step))
