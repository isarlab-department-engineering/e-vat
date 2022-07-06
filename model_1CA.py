import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturesExtractor_RGB(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_RGB, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.width = settings['width']
        self.height = settings['height']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=8, stride=4, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 8 * 8

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, self.channels, self.width, self.height)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5

    @property
    def device(self):
        return next(self.parameters()).device


class FeaturesExtractor_Grid(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_Grid, self).__init__()
        self.settings = settings
        self.grid = settings['grid_cells']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d1,
                               kernel_size=4, stride=2, padding=2)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 7 * 7

    def forward(self, input_data):
        inp = input_data['bel'].reshape(-1, 3, self.grid, self.grid).to(self.device)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5

    @property
    def device(self):
        return next(self.parameters()).device


class FeaturesExtractor_Ego(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_Ego, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.width = 21
        self.height = 11
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1,
                               kernel_size=3, stride=2, padding=1)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d1,
                               kernel_size=3, stride=1, padding=1)
        self.bnc2 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv3 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=1)
        self.bnc3 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)
        self.conv4 = nn.Conv2d(in_channels=self.d2, out_channels=self.d2,
                               kernel_size=3, stride=1, padding=0)
        self.bnc4 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 9 * 4

    def forward(self, input_data):
        inp = input_data['ego_target'].reshape(-1, self.channels, self.width, self.height)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))
        x4 = self.bnc3(F.relu(self.conv3(x3)))
        x5 = self.bnc4(F.relu(self.conv4(x4)))

        return x5

    @property
    def device(self):
        return next(self.parameters()).device
######################################################################################################################


# ###################################-Tracking+Exploration-###########################################################
class ActorIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.RGB_shape = self.features_RGB.feat_shape

        # Linear
        self.lin1 = nn.Linear(self.RGB_shape, 750)
        self.lin2 = nn.Linear(750, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # #################-tail_track-#############################
        self.hidden0_track = nn.Linear(self.dr, 256)
        self.hidden1_track = nn.Linear(256, 128)
        self.output_track = nn.Linear(128, self.settings['output_a'])

        # ##################-tail_exp-##############################
        self.hidden0_exp = nn.Linear(self.dr, 256)
        self.hidden1_exp = nn.Linear(256, 128)
        self.output_exp = nn.Linear(128, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['track_h_a']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)

        # lin
        x1 = F.relu(self.lin1(featuresRGB))
        x2 = F.relu(self.lin2(x1))

        # gru
        x_ = x2.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)

        # #################-tail_track-#############################
        xh0_track = F.relu(self.hidden0_track(xgru))
        xh1_track = F.relu(self.hidden1_track(xh0_track))
        logits_track = self.output_track(xh1_track)

        # ##################-tail_exp-##############################
        xh0_exp = F.relu(self.hidden0_exp(xgru))
        xh1_exp = F.relu(self.hidden1_exp(xh0_exp))
        logits_exp = self.output_exp(xh1_exp)

        return logits_track, logits_exp, h_a_


class FOV_Net(nn.Module):
    def __init__(self, settings):
        super(FOV_Net, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.RGB_shape = self.features_RGB.feat_shape

        # Linear
        self.lin1 = nn.Linear(self.RGB_shape, 750)
        self.lin2 = nn.Linear(750, 256)
        self.lin3 = nn.Linear(256, 128)
        self.output_fov = nn.Linear(128, 1)

    def forward(self, input_data):
        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)

        # lin
        x1 = F.relu(self.lin1(featuresRGB))
        x2 = F.relu(self.lin2(x1))
        x3 = F.relu(self.lin3(x2))
        fov = self.output_fov(x3)

        return fov


class CriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(CriticIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.lin_dis_angle_hit = nn.Linear(3, 300)
        self.features_Grid = FeaturesExtractor_Grid(settings)
        self.Grid_shape = self.features_Grid.feat_shape
        self.features_Ego = FeaturesExtractor_Ego(settings)
        self.Ego_shape = self.features_Ego.feat_shape
        self.lin_yaw = nn.Linear(1, 100)
        # -
        self.feat_shape = self.Grid_shape + self.Ego_shape
        self.lin = nn.Linear(self.feat_shape + 100 + 300, self.dl)
        self.gru = nn.GRU(self.dl, self.dr)
        # hidden
        self.hidden1 = nn.Linear(self.dr, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        # track_h_c equal to exp_h_c
        input_track_hc = input_data['track_h_c']
        input_exp_hc = input_data['exp_h_c']
        input_yaw = input_data['yaw'].reshape(-1, 1)
        distance = input_data['distance'].reshape(-1, 1)
        angle = input_data['angle'].reshape(-1, 1)
        hit = input_data['hit'].reshape(-1, 1)
        featuresGrid = self.features_Grid.forward(input_data).reshape(-1, self.Grid_shape)
        featuresEgo = self.features_Ego.forward(input_data).reshape(-1, self.Ego_shape)

        dis_angle_hit = torch.cat((distance, angle, hit), dim=1)
        x_dist_angle_hit = F.relu(self.lin_dis_angle_hit(dis_angle_hit))
        features = torch.cat((featuresGrid, featuresEgo), dim=1)
        x_yaw = F.relu(self.lin_yaw(input_yaw))
        x_cat = torch.cat((features, x_yaw, x_dist_angle_hit), dim=1)
        x = F.relu(self.lin(x_cat))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, exp_h_c_ = self.gru(x_, input_exp_hc)
        xh1 = F.relu(self.hidden1(xgru))
        xh2 = F.relu(self.hidden2(xh1))
        values_exp = self.output(xh2)

        return values_exp, values_exp, exp_h_c_, exp_h_c_


# #######################################-Impala_Net-###################################################################
class IMPALA_Net(nn.Module):
    def __init__(self, settings):
        super(IMPALA_Net, self).__init__()
        self.settings = settings

        self.actor = ActorIMPALA(settings)
        self.fov_net = FOV_Net(settings)
        self.critic = CriticIMPALA(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        logits_track, logits_exp, h_a_ = self.actor(input_data)
        fov = self.fov_net(input_data)
        values_track, values_exp, track_h_c_, exp_h_c_ = self.critic(input_data)
        output_data = {'logits_track': logits_track,
                       'logits_exp': logits_exp,
                       'values_track': values_track,
                       'values_exp': values_exp,
                       'track_h_a_': h_a_, 'exp_h_a_': h_a_, 'fov_h_a_': h_a_,
                       'track_h_c_': track_h_c_, 'exp_h_c_': exp_h_c_,
                       'fov': fov}

        return output_data

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        input_data['track_h_a'] = torch.from_numpy(input_data['track_h_a']).float().to(input_data['device'])
        input_data['exp_h_a'] = torch.from_numpy(input_data['exp_h_a']).float().to(input_data['device'])
        input_data['fov_h_a'] = torch.from_numpy(input_data['fov_h_a']).float().to(input_data['device'])
        input_data['track_h_c'] = torch.from_numpy(input_data['track_h_c']).float().to(input_data['device'])
        input_data['exp_h_c'] = torch.from_numpy(input_data['exp_h_c']).float().to(input_data['device'])

        forward_out = self.forward(input_data)

        logits_track = forward_out['logits_track']
        probs_track = torch.clamp(F.softmax(logits_track, dim=-1), 0.00001, 0.99999).data  # .values()
        m_track = torch.distributions.Categorical(probs_track)
        action_track = m_track.sample().type(torch.IntTensor)

        logits_exp = forward_out['logits_exp']
        probs_exp = torch.clamp(F.softmax(logits_exp, dim=-1), 0.00001, 0.99999).data  # .values()
        m_exp = torch.distributions.Categorical(probs_exp)
        action_exp = m_exp.sample().type(torch.IntTensor)

        output_data = {'action_track': action_track.detach().cpu().numpy().squeeze(),
                       'logits_track': logits_track.detach().squeeze().cpu().numpy(),
                       'action_exp': action_exp.detach().cpu().numpy().squeeze(),
                       'logits_exp': logits_exp.detach().squeeze().cpu().numpy(),
                       'track_h_a_': forward_out['track_h_a_'],
                       'exp_h_a_': forward_out['exp_h_a_'],
                       'fov_h_a_': forward_out['fov_h_a_'],
                       'track_h_c_': forward_out['track_h_c_'],
                       'exp_h_c_': forward_out['exp_h_c_'],
                       'fov': forward_out['fov']}

        return output_data

    def get_values(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)
        values_out = {'values_track': forward_out['values_track'],
                      'values_exp': forward_out['values_exp']}
        return values_out

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h
########################################################################################################################
