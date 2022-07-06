import time
import torch
import torch.nn.functional as F
import numpy as np
import gym
import gym_unrealisar  # TODO: replace with your own gym and register your envs
import copy


def create_env(settings, seed=10):
    # print(settings['env_name'])
    env = gym.make(settings['env_name'])
    env.action_space = settings['action_space']
    env.seed(seed)
    print('Create env')
    env.connect_unreal(settings['unreal']['ports'], settings['unreal']['json_paths'],
                       render=settings['unreal']['render'], address=settings['unreal']['address'])
    env.test = settings['test']
    if not settings['IMPALA']:
        env = NormalizedEnv(env)  # crop action space
    return env


def get_input_data_supervised(samples, device):
    RGBCamera = samples['RGBCamera']
    angle = samples['angle']
    distance = samples['distance']
    x = samples['x']
    y = samples['y']
    yaw = samples['yaw']

    # from numpy to tensor
    RGBCamera = torch.from_numpy(RGBCamera).float().to(device)
    angle = torch.from_numpy(angle).float().to(device)
    distance = torch.from_numpy(distance).float().to(device)
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    yaw = torch.from_numpy(yaw).float().to(device)

    input_data = dict()
    input_data['get_action'] = {'RGBCamera': RGBCamera, 'angle': angle, 'distance': distance, 'x': x, 'y': y,
                                'yaw': yaw}

    return input_data


def get_input_data_IMPALA(samples, device):
    states = samples['RGBCamera']
    actions_track = samples['action_track']
    actions_exp = samples['action_exp']
    rewards_track = samples['reward_track']
    rewards_exp = samples['reward_exp']
    done_track = samples['done_track']
    done_exp = samples['done_exp']
    track_h_a = samples['track_h_a']
    exp_h_a = samples['exp_h_a']
    fov_h_a = samples['fov_h_a']
    track_h_c = samples['track_h_c']
    exp_h_c = samples['exp_h_c']
    logits_track = samples['logits_track']
    logits_exp = samples['logits_exp']
    states_ = samples['next_state']
    ego_target = samples['ego_target']
    yaw = samples['yaw']
    angle = samples['angle']
    distance = samples['distance']
    hit = samples['hit']
    bel = samples['bel']
    fov_gt = samples['fov_gt']
    track_flag = samples['track_flag']

    states = torch.from_numpy(states).float().to(device)
    actions_track = torch.from_numpy(actions_track).float().to(device)
    actions_exp = torch.from_numpy(actions_exp).float().to(device)
    rewards_track = torch.from_numpy(rewards_track).float().to(device)
    rewards_exp = torch.from_numpy(rewards_exp).float().to(device)
    done_track = torch.from_numpy(done_track).float().to(device)
    done_exp = torch.from_numpy(done_exp).float().to(device)
    track_h_a = torch.from_numpy(track_h_a).float().to(device)
    exp_h_a = torch.from_numpy(exp_h_a).float().to(device)
    fov_h_a = torch.from_numpy(fov_h_a).float().to(device)
    track_h_c = torch.from_numpy(track_h_c).float().to(device)
    exp_h_c = torch.from_numpy(exp_h_c).float().to(device)
    logits_track = torch.from_numpy(logits_track).float().to(device)
    logits_exp = torch.from_numpy(logits_exp).float().to(device)
    states_ = torch.from_numpy(states_).float().to(device)
    stack_states_ = torch.cat((states[1:], states_))
    ego_target = torch.from_numpy(ego_target).float().to(device)
    yaw = torch.from_numpy(yaw).float().to(device)
    angle = torch.from_numpy(angle).float().to(device)
    distance = torch.from_numpy(distance).float().to(device)
    hit = torch.from_numpy(hit).float().to(device)
    fov_gt = torch.from_numpy(fov_gt).float().to(device)
    track_flag = torch.from_numpy(track_flag).to(device)
    bel = torch.from_numpy(bel).float().to(device)

    input_data = dict()
    input_data['RGBCamera'] = states
    input_data['action_track'] = actions_track
    input_data['action_exp'] = actions_exp
    input_data['reward_track'] = rewards_track.squeeze()
    input_data['reward_exp'] = rewards_exp.squeeze()
    input_data['new_state'] = stack_states_
    input_data['done_track'] = done_track
    input_data['done_exp'] = done_exp
    input_data['track_h_a'] = track_h_a.squeeze_().unsqueeze_(0)
    input_data['exp_h_a'] = exp_h_a.squeeze_().unsqueeze_(0)
    input_data['fov_h_a'] = fov_h_a.squeeze_().unsqueeze_(0)
    input_data['track_h_c'] = track_h_c.squeeze_().unsqueeze_(0)
    input_data['exp_h_c'] = exp_h_c.squeeze_().unsqueeze_(0)
    input_data['logits_track'] = logits_track
    input_data['logits_exp'] = logits_exp
    input_data['yaw'] = yaw.reshape(-1)
    input_data['angle'] = angle.reshape(-1)
    input_data['distance'] = distance.reshape(-1)
    input_data['hit'] = hit.reshape(-1)
    input_data['fov_gt'] = fov_gt
    input_data['ego_target'] = ego_target
    input_data['track_flag'] = track_flag
    input_data['bel'] = bel

    return input_data


class UniformNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.5, decay_period=50000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        pass

    def get_action(self, action, t=0):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        noised_action = action + self.sigma * np.random.randn(self.action_dim)
        return np.clip(noised_action, self.low, self.high)[0]


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)[0]


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min,
                                                       n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class NormalizedEnv1(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        tracker_actions = np.clip(action, -1, 1)

        act_k0 = (self.action_space[0].high - self.action_space[0].low) / 2.
        act_b0 = (self.action_space[0].high + self.action_space[0].low) / 2.

        act_k1 = (self.action_space[1].high - self.action_space[1].low) / 2.
        act_b1 = (self.action_space[1].high + self.action_space[1].low) / 2.
        return [(act_k0 * tracker_actions[0] + act_b0)[0], (act_k1 * tracker_actions[1] + act_b1)[0]]

    def reverse_action(self, action):
        tracker_actions = action[0]
        target_action = action[1]
        act_k_inv0 = 2. / (self.action_space[0].high - self.action_space[0].low)
        act_b0 = (self.action_space[0].high + self.action_space[0].low) / 2.

        act_k_inv1 = 2. / (self.action_space[1].high - self.action_space[1].low)
        act_b1 = (self.action_space[1].high + self.action_space[1].low) / 2.

        return [[(act_k_inv0 * (tracker_actions[0] - act_b0))[0], (act_k_inv1 * (tracker_actions[1] - act_b1))[0]],
                target_action]


class Belief(object):
    def __init__(self, settings, device):
        self.settings = settings
        self.device = device

    def init_target_prob(self, input_grid):
        input_grid = input_grid.float()
        free_cells = input_grid[2].nelement() - len(torch.nonzero(input_grid[0].reshape(-1, 1)))
        input_grid[2] = torch.where(input_grid[0] > 0, torch.tensor([0.]).to(self.device),
                                    torch.tensor([1.]).to(self.device) / torch.tensor(free_cells).to(self.device))
        return input_grid

    def update_target_prob(self, input_grid, ego_hat, tracker_p, yaw, sensor_map, gt=True):
        if not gt:
            sensor_map = self.get_geo_target_sensor(ego_hat.unsqueeze(0), tracker_p, yaw).squeeze()
        measurement_t_t, measurement_t_nt = self.measurement_model(sensor_map, input_grid[0])
        bel_t = torch.ones_like(input_grid[2]) * input_grid[2] * measurement_t_t
        bel_nt = (torch.ones_like(input_grid[2]) - input_grid[2]) * measurement_t_nt

        bel = bel_t + bel_nt
        bel = bel_t * bel.pow(-1)

        bel = torch.where(input_grid[0] > 0., torch.tensor([0.0]).to(self.device), bel)

        input_grid[2] = bel * torch.sum(bel).pow(-1)

        return input_grid

    def measurement_model(self, sensor_map, obstacle_map):
        sensor_map = sensor_map.float()
        p_target_target = 0.9
        p_target_no_target = 0.1
        tracker_confidence = torch.tensor([0.9]).to(self.device)
        sensor_thr = 0.1
        grid = torch.zeros_like(sensor_map[2]).float().to(self.device)

        grid_t_t = torch.where(sensor_map[2] > 0.8, torch.tensor([0.01]).to(self.device), grid).to(self.device)
        grid_t_nt = torch.where(sensor_map[2] > 0.8, torch.tensor([0.99]).to(self.device), grid).to(self.device)

        grid_t_t = torch.where(sensor_map[0] > 0, torch.tensor([p_target_target]).to(self.device), grid_t_t)
        grid_t_nt = torch.where(sensor_map[0] > 0, torch.tensor([p_target_no_target]).to(self.device), grid_t_nt)

        grid_t_t = torch.where(grid_t_t == 0.0, torch.tensor([0.5]).to(self.device), grid_t_t).to(self.device)
        grid_t_nt = torch.where(grid_t_nt == 0.0, torch.tensor([0.5]).to(self.device), grid_t_nt).to(self.device)

        p = torch.clamp(grid_t_t, min=0., max=255.).detach().unsqueeze(-1).cpu().numpy()
        p = np.concatenate((p, np.zeros_like(p), np.zeros_like(p)), axis=2)
        # img = cv.cvtColor(p, cv.COLOR_BGR2RGB)
        # if self.settings['eval_mode']:
        #     cv.imshow('mask', p)
        #     cv.waitKey(1)

        return grid_t_t, grid_t_nt

    def get_geo_target_sensor(self, ego_grid_hat, tracker_p, yaw):
        rec_grid = torch.zeros(1, 3, self.settings['model_param']['grid_cells'], self.settings['model_param']['grid_cells']).to(self.device)
        # turn back
        angle = torch.from_numpy(np.array(np.radians(yaw.cpu().numpy()))).to(self.device)
        theta = torch.zeros(rec_grid.size(0), 2, 3).to(self.device)
        if angle.nelement() > 1:
            for i in range(len(angle)):
                theta[i, :, :2] = torch.tensor([[torch.cos(angle[i]).to(self.device), torch.tensor([-1.0]).to(self.device) * torch.sin(angle[i]).to(self.device)],
                                                [torch.sin(angle[i]).to(self.device), torch.cos(angle[i]).to(self.device)]])
        else:
            theta[:, :, :2] = torch.tensor([[torch.cos(angle).to(self.device), torch.tensor([-1.0]).to(self.device) * torch.sin(angle).to(self.device)],
                                            [torch.sin(angle).to(self.device), torch.cos(angle).to(self.device)]])

        ego_grid_hat = F.pad(input=ego_grid_hat[:], pad=[0, 0, 0, ego_grid_hat.size(-1) - ego_grid_hat.size(-2)], mode='constant', value=0)  # [-1, 3, 21, 21]
        grid = F.affine_grid(theta, ego_grid_hat.size())
        x_trans = F.grid_sample(ego_grid_hat, grid)
        grid_affine = x_trans

        # geocentric grid reconstruction
        pad = int(grid_affine.size(-1) / 2) + 1
        rec_grid = F.pad(input=rec_grid[:], pad=[pad, pad, pad, pad], mode='constant', value=0)
        for i in range(tracker_p.size(0)):
            tracker_pos = (int(tracker_p[i, 0] + pad), int(tracker_p[i, 1] + pad))

        rec_grid[:, :, tracker_pos[0] - pad + 1: tracker_pos[0] + pad, tracker_pos[1] - pad + 1: tracker_pos[1] + pad] += grid_affine[:, :]

        rec_grid = rec_grid[:, :, pad: -pad, pad: -pad]

        return rec_grid


class ScoreCounter:
    def __init__(self, dis_exp, dis_max):
        self.scores = {'score_angle_total': 0., 'score_angle_track': 0., 'score_angle_effective': 0.,
                       'score_distance_total': 0., 'score_distance_track': 0., 'score_distance_effective': 0.,
                       'score_perc_total': 0., 'score_perc_track': 0., 'score_perc_effective': 0., 'coverage_perc': 0.,
                       'target_found': 0, 'target_lost': 0, 'target_lost_effective': 0, 'target_recovered': 0}
        self.check_list = {'total_steps': 0, 'track_steps': 0, 'effective_track_steps': 0, 'exp_steps': 0,
                           'episodes': 0, 'dis_exp': dis_exp, 'dis_max': dis_max}
        self.exploration_scores = {'distance_traveled': [], 'target_found_episodes': [], 'exp_steps_episodes': []}
        self.dis_exp = dis_exp
        self.dis_max = dis_max
        self.target_found = False
        self.target_lost = False
        self.target_lost_effective = False
        self.coverage_best_episode = 0

        # Exploration Rebuttal
        self.last_position = None
        self.distance_traveled = 0
        self.effective_exp_steps = 0

    def add_score(self, info):
        self.check_list['total_steps'] += 1
        distance_to_target = info['distance']
        direction_error = info['angle']
        hit = info['hit']
        geo_grid = info['Geo_target']
        x = copy.deepcopy(info['GPS_X'])
        y = copy.deepcopy(info['GPS_Y'])
        if self.last_position is None:
            self.last_position = [x, y]
        if not self.target_found:
            self.distance_traveled += np.sqrt((self.last_position[0] - x)**2 + (self.last_position[1] - y)**2)
            self.last_position = [x, y]

        (_, r, c) = geo_grid.shape
        exp_perc = (((r * c) - np.sum(geo_grid[0])) - np.sum(geo_grid[2])) / ((r * c) - np.sum(geo_grid[0]))
        if self.coverage_best_episode < exp_perc:
            self.coverage_best_episode = exp_perc

        distance_error = abs(distance_to_target - self.dis_exp) / self.dis_max

        direction_error = abs(direction_error / 40.0)

        score_distance = max(1 - distance_error, 0) if direction_error < 1 and hit else 0
        score_angle = max(1 - direction_error, 0) if distance_to_target < 130 and hit else 0
        score_total = score_distance + score_angle
        score_perc = 1 if score_total > 0 else 0

        if score_total > 0:
            self.check_list['track_steps'] += 1
            self.target_found = True
            if self.target_lost:
                self.scores['target_recovered'] += 1
            self.target_lost = False
            self.scores['score_angle_track'] += round(score_angle, 3)
            self.scores['score_distance_track'] += round(score_distance, 3)
            self.scores['score_perc_track'] += round(score_perc, 3)
        else:
            self.check_list['exp_steps'] += 1
            if self.target_found:
                self.target_lost = True
                self.target_lost_effective = True
            else:
                self.effective_exp_steps += 1

        if self.target_found:
            self.check_list['effective_track_steps'] += 1
            self.scores['score_angle_effective'] += round(score_angle, 3)
            self.scores['score_distance_effective'] += round(score_distance, 3)
            self.scores['score_perc_effective'] += round(score_perc, 3)

        self.scores['score_angle_total'] += round(score_angle, 3)
        self.scores['score_distance_total'] += round(score_distance, 3)
        self.scores['score_perc_total'] += round(score_perc, 3)

    def reset(self):
        if self.target_found:
            self.scores['target_found'] += 1
            self.exploration_scores['target_found_episodes'].append(1)
        else:
            self.exploration_scores['target_found_episodes'].append(0)
        if self.target_lost:
            self.scores['target_lost'] += 1
        if self.target_lost_effective:
            self.scores['target_lost_effective'] += 1
        self.check_list['episodes'] += 1
        print('\nepisode: {}\ntarget_found: {}\ntarget_lost: {}\ntarget_lost_effective: {}\ntarget_recovered: {}'.
              format(self.check_list['episodes'], self.scores['target_found'], self.scores['target_lost'],
                     self.scores['target_lost_effective'], self.scores['target_recovered']))

        self.scores['coverage_perc'] = round((self.scores['coverage_perc'] * (self.check_list['episodes'] - 1) +
                                              self.coverage_best_episode) / self.check_list['episodes'], 3)
        self.coverage_best_episode = 0

        self.exploration_scores['exp_steps_episodes'].append(self.effective_exp_steps)
        self.exploration_scores['distance_traveled'].append(self.distance_traveled)
        self.target_found = False
        self.target_lost = False
        self.target_lost_effective = False
        self.effective_exp_steps = 0
        self.last_position = None
        self.distance_traveled = 0

    def save_scores(self, ep, st, tm, name):
        print(self.scores)
        timestr = time.strftime('%d%m%Y-%H%M')
        f = open('C:\\Users\\Perseo\\Documents\\ActiveTracking\\experiments\\rebuttal\\exp_1CA_evat_' + timestr + '.txt', 'w')
        f.write(str(self.exploration_scores))
        f.write('\n\n')
        for k, v in self.scores.items():
            f.write('{}'.format(k.ljust(25)))
        for k, v in self.check_list.items():
            f.write('{}'.format(k.ljust(25)))
        f.write('\n\n')
        for k, v in self.scores.items():
            f.write('{}'.format(str(round(v, 3)).ljust(25)))
        for k, v in self.check_list.items():
            f.write('{}'.format(str(round(v, 3)).ljust(25)))
        f.write('\n\n')

        f.write('Score Angle Mean Total:    {:<20}Score Angle Mean Track:    {:<20}'
                'Score Angle Mean Effective Track:    {:<20}\n\n'.format(
                 round(self.scores['score_angle_total']/self.check_list['total_steps'], 3),
                 round(self.scores['score_angle_track']/self.check_list['track_steps'], 3)
                       if self.check_list['track_steps'] > 0 else 0.,
                 round(self.scores['score_angle_effective']/self.check_list['effective_track_steps'], 3)
                       if self.check_list['effective_track_steps'] > 0 else 0.))

        f.write('Score Distance Mean Total: {:<20}Score Distance Mean Track: {:<20}'
                'Score Distance Mean Effective Track: {:<20}\n\n'.format(
                 round(self.scores['score_distance_total'] / self.check_list['total_steps'], 3),
                 round(self.scores['score_distance_track'] / self.check_list['track_steps'], 3)
                       if self.check_list['track_steps'] > 0 else 0.,
                 round(self.scores['score_distance_effective'] / self.check_list['effective_track_steps'], 3)
                       if self.check_list['effective_track_steps'] > 0 else 0.))

        f.write('Score Total Mean Total:    {:<20}Score Total Mean Track:    {:<20}'
                'Score Total Mean Effective Track:    {:<20}\n\n'.format(
                 round((self.scores['score_angle_total'] + self.scores['score_distance_total']) / self.check_list['total_steps'] / 2, 3),
                 round((self.scores['score_angle_track'] + self.scores['score_distance_track']) / self.check_list['track_steps'] / 2, 3)
                       if self.check_list['track_steps'] > 0 else 0.,
                 round((self.scores['score_angle_effective'] + self.scores['score_distance_effective']) / self.check_list['effective_track_steps'] / 2, 3)
                       if self.check_list['effective_track_steps'] > 0 else 0.))

        f.write('Score Perc Mean Total:     {:<20}Score Perc Mean Track:     {:<20}'
                'Score Perc Mean Effective Track:     {:<20}\n\n'.format(
                 round(self.scores['score_perc_total'] / self.check_list['total_steps'], 3),
                 round(self.scores['score_perc_track'] / self.check_list['track_steps'], 3)
                       if self.check_list['track_steps'] > 0 else 0.,
                 round(self.scores['score_perc_effective'] / self.check_list['effective_track_steps'], 3)
                       if self.check_list['effective_track_steps'] > 0 else 0.))

        f.write('model name:         {}\n'
                'number of episodes: {}\n'
                'episodes length:    {}\n'
                'elapsed time:       {}m'
                .format(name, ep, st, round(tm/60.), 1))

        f.close()
