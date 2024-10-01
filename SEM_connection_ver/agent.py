import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACNet(nn.Module):
    def __init__(self, state_size, action_dim):
        super(ACNet, self).__init__()
        self.l1 = nn.Linear(state_size, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, action_dim)

        self.b1 = nn.BatchNorm1d(512)
        self.b2 = nn.BatchNorm1d(512)
        self.b3 = nn.BatchNorm1d(512)
        self.b4 = nn.BatchNorm1d(512)

    def forward(self, state):
        a = F.leaky_relu(self.b1(self.l1(state)), 0.2)
        a = F.leaky_relu(self.b2(self.l2(a)), 0.2)
        a = F.leaky_relu(self.b3(self.l3(a)), 0.2)
        a = F.leaky_relu(self.b4(self.l4(a)), 0.2)
        a = self.l5(a)
        a = torch.tanh(a).float()
        return a.squeeze()

class Agent(object):
    def __init__(self, env, state_size, action_dim, replay_buffer):
        self.env = env
        self.actor = ACNet(state_size, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.actor_criterion = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=150, gamma=0.1)
        self.max_action = 0
        self.replay_buffer = replay_buffer
        self.noise = 0

    def set_max_action(self, max_action):
        self.max_action = max_action

    def select_action(self, state, noise):
        self.actor.eval()
        self.noise = noise
        state = state.to(device)
        state = torch.reshape(state, [1, state.size()[0]])
        action = (self.max_action * self.actor(state)).cpu().data.numpy().flatten()

        if self.noise != 0:
            noise = np.random.normal(0, self.noise, size=1)
            if self.max_action < 1.5:
                noise = noise / 100
            action = action + noise
            action = action.clip(-self.max_action, self.max_action)

        return torch.tensor([action], device=device, dtype=torch.float)

    def store_transition(self, state, action, target_action, done):
        state = torch.reshape(state, [1, state.size()[0]])
        target_action = target_action.reshape(1, -1) / self.max_action
        action = action.reshape(1, -1)
        self.replay_buffer.add(state, action, target_action, done)

    def train(self, replay_buffer):
        self.actor.train()
        # Sample replay buffer
        states, actions, target_actions, dones = replay_buffer.sample()

        # Select action according to network
        outputs = self.actor(states)

        self.actor_optimizer.zero_grad()

        # Get Actor's loss using MSE criterion
        loss = self.actor_criterion((1 - dones) * outputs.squeeze(), (1 - dones) * target_actions.squeeze())
        loss.backward()
        self.actor_optimizer.step()

        accuracy = 0
        for i in range(0, len(outputs)):
            accuracy += 1 - dones[i].item() * (np.abs(outputs[i].item() - target_actions[i].item()))
        accuracy /= len(outputs)

        # self.scheduler.step()
        return loss, accuracy

    def save(self, episode, filename):
        if not os.path.exists('./Trained_network/re_train/' + filename):
            os.makedirs('./Trained_network/re_train/' + filename)

        actor_saving_dict = {
            'Episode': episode,
            'State_dict': self.actor.state_dict(),
            'optimizer': self.actor_optimizer.state_dict()
        }
        torch.save(actor_saving_dict, './Trained_network/re_train/%s/actor.pth' % (filename))

    def load(self):
        if os.path.exists('./Trained_network/SL_actor_action_size_random_init_210125.pth'):
            checkpoint = torch.load('./Trained_network/SL_actor_action_size_random_init_210125.pth', map_location='cuda:0')
            self.actor.load_state_dict(checkpoint['State_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer'])

            print('saved state dict of Actor is loaded.')

            episode = checkpoint['Episode']
        else:
            raise RuntimeError("Can not load trained actor.")                
