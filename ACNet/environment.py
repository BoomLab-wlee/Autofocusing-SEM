import simulation_client
import random
import torch
import numpy as np
import math
import time

class env():
    def __init__(self, mode='train'):
        self.mode = mode
        self.device = torch.device('cuda')
        self.client = simulation_client.Simulation_Client()
        self.sample = 0
        self.frame_num = 20
        self.state = []
        # self.state_size = (self.frame_num - 1) + (self.frame_num * 3) + 1
        # self.state_size = (self.frame_num - 1) + (self.frame_num * 2) + 1
        self.state_size = (self.frame_num - 1) + (self.frame_num * 1) + 1
        self.sequence = ''
        self.action_dim = 1
        self.max_action = 1.2
        self.init_wd = 0.001
        self.next_wd = 0.001
        self.scale = 1
        self.target_action = 0
        self.done = 0
        self.iteration = 0
        self.iter_max = 30
        self.target = 0
        self.set_mag = 0
        self.memory = []
        self.max_wd = 0
        self.noise = 0

    def action_space(self):
        '''Define SEM client action dimension'''

        return self.state_size, self.action_dim

    def reset(self, sample=1):
        if self.mode == 'train' or self.mode == 're-train':
            self.sample = random.randrange(1, 92)
            self.sequence = random.choice(['Forward', 'Forward', 'Inverse'])
            self.noise = random.randrange(0, 21) / 100
        elif self.mode == 'test':
            self.sample = sample
            self.sequence = 'Forward'
            self.noise = 0
            self.init_wd = 0.001

        self.client.send('reset', self.sample)
        self.client.send('noise', self.noise)

        self.target = torch.tensor(self.client.target, device=self.device, dtype=torch.float)

        if self.mode == 'train':
            if self.sequence == 'Forward':
                self.init_wd = random.randrange(1, self.target*1000+1)/1000
            elif self.sequence == 'Inverse':
                self.init_wd = random.randrange(self.target*1000, self.client.data_length+1)/1000
        elif self.mode == 're-train':
            if self.sequence == 'Forward':
                self.init_wd = random.randrange(1, self.target*1000+1)/1000
            elif self.sequence == 'Inverse':
                self.init_wd = random.randrange(self.target*1000, self.client.data_length+1)/1000

        self.max_action = 1.2
        self.scale = 1

        self.client.send('WD', self.init_wd)

        self.memory = torch.zeros([self.frame_num, 5], device=self.device)
        self.state = torch.zeros([1, self.state_size], device=self.device)

        score, var, ent, mag, done = self.client.receive()
        self.set_mag = mag

        self.set_action_scale(mag)

        self.iteration = 0
        self.check = 0

        self.initial_searching()

        return self.state, self.max_action

    def initial_searching(self):
        if self.mode == 'train':
            init_stride = random.randrange(3, 6)/10
            init_iter_max = random.randrange(1, 6)
        elif self.mode == 'test':
            init_stride = 0.6
            init_iter_max = 1

        stride = np.round(init_stride * self.scale, 3)

        if self.sequence == 'Forward':
            stride = stride
        elif self.sequence == 'Inverse':
            stride = -stride

        wd2, score2, var2, ent2, mag2 = torch.zeros([5], device=self.device)
        wd3, score3, var3, ent3, mag3 = torch.zeros([5], device=self.device)

        while True:
            self.iteration += 1
            self.next_wd = self.init_wd + stride*self.iteration
            self.client.send('WD', self.next_wd)

            score, var, ent, mag, done = self.client.receive()

            if self.iteration > init_iter_max and score > 2.0:
                print('Initial_processing is done, Init iteration: {}'.format(self.iteration))
                print('[Sample= {}, Mag= {}] WD: {:.3f}, Score: {:.2f}'.format(self.sample, self.set_mag*100000, np.round(self.next_wd, 3), np.round(score.item(), 2)))
                self.iteration = 0
                break
            elif self.next_wd + stride > self.client.data_length/1000 or self.next_wd + stride < 0.001:
                print('Initial_processing is done, Init iteration: {}'.format(self.iteration))
                print('[Sample= {}, Mag= {}] WD: {:.3f}, Score: {:.2f}'.format(self.sample, self.set_mag*100000, np.round(self.next_wd, 3), np.round(score.item(), 2)))
                self.iteration = 0
                break
            else:
                wd3, score3, var3, ent3, mag3 = wd2, score2, var2, ent2, mag2
                wd2, score2, var2, ent2, mag2 = self.next_wd, score, var, ent, mag

        wd3, score3, var3, ent3, mag3 = self.state_scaling(wd3, score3, var3, ent3, mag3)
        wd2, score2, var2, ent2, mag2 = self.state_scaling(wd2, score2, var2, ent2, mag2)

        self.state_memory(wd3, score3, var3, ent3, mag3)
        self.state_memory(wd2, score2, var2, ent2, mag2)
        self.set_state(self.next_wd, score, var, ent, mag)

    def step(self, action):
        self.iteration += 1

        action = action.item()
        self.next_wd = np.round(action + self.next_wd, 3)

        self.client.send('WD', self.next_wd)
        # time.sleep(0.005)

        score, var, ent, mag, done = self.client.receive()

        if done != 1:
            self.set_state(self.next_wd, score, var, ent, mag)

            # give reward and done value
            self.get_optimal_action(action)
            self.terminal_condition()

            print( '[Sample= {}, Mag= {}] WD: {:.3f}, Score: {:.2f}, Action: {:.4f}, Target_Action: {:.4f}'
                .format(self.sample, self.set_mag*100000, np.round(self.next_wd, 3), np.round(score.cpu().detach(), 2), np.round(action, 3), np.round(self.target_action.item(), 3)))

        elif done == 1:
            self.done = torch.tensor([1], device=self.device, dtype=torch.float)
            self.target_action = torch.tensor([1], device=self.device, dtype=torch.float)
        return self.state, self.target_action, self.done, self.next_wd

    def get_optimal_action(self, action):
        if (self.target - self.memory[0, 0]*self.max_action)*(self.target - self.memory[1, 0]*self.max_action) == 0:
            if self.target - self.memory[1, 0]*self.max_action == 0:
                target_action = 0
            elif self.target - self.memory[0, 0]*self.max_action == 0:
                target_action = action
        else:
            target_action = (self.target - self.memory[1, 0]*self.max_action).cpu().data.numpy().flatten()
            target_action = target_action.clip(-self.max_action, self.max_action)

        self.target_action = torch.tensor([target_action], device=self.device, dtype=torch.float)

    def state_memory(self, wd, score, var, ent, mag):
        if torch.sum(self.memory).item() == 0:
            for i in range(0, self.frame_num-1):
                self.memory[i, :] = torch.stack([wd, score, var, ent, mag])

        for i in range(0, self.frame_num-1):
            i = -i+(self.frame_num-2)
            self.memory[i+1, :] = self.memory[i, :]
        self.memory[0, :] = torch.stack([wd, score, var, ent, mag])

    def state_scaling(self, wd, score, var, ent, mag):
        wd = torch.tensor(wd, device=self.device)/self.max_action
        score = score/10
        var = var
        ent = ent/100
        mag = mag
        return wd, score, var, ent, mag

    def set_state(self, wd, score, var, ent, mag):
        wd, score, var, ent, mag = self.state_scaling(wd, score, var, ent, mag)
        self.state_memory(wd, score, var, ent, mag)

        # self.state = torch.stack(
        #     [wd - self.memory[1, 0], wd - self.memory[2, 0], wd - self.memory[3, 0], wd - self.memory[4, 0],
        #      wd - self.memory[5, 0], wd - self.memory[6, 0], wd - self.memory[7, 0], wd - self.memory[8, 0],
        #      wd - self.memory[9, 0], wd - self.memory[10, 0], wd - self.memory[11, 0], wd - self.memory[12, 0],
        #      wd - self.memory[13, 0], wd - self.memory[14, 0], wd - self.memory[15, 0], wd - self.memory[16, 0],
        #      wd - self.memory[17, 0], wd - self.memory[18, 0], wd - self.memory[19, 0],
        #      score, score - self.memory[1, 1], score - self.memory[2, 1], score - self.memory[3, 1],
        #      score - self.memory[4, 1], score - self.memory[5, 1], score - self.memory[6, 1],
        #      score - self.memory[7, 1], score - self.memory[8, 1], score - self.memory[9, 1],
        #      score - self.memory[10, 1], score - self.memory[11, 1], score - self.memory[12, 1],
        #      score - self.memory[13, 1], score - self.memory[14, 1], score - self.memory[15, 1],
        #      score - self.memory[16, 1], score - self.memory[17, 1], score - self.memory[18, 1],
        #      score - self.memory[19, 1],
        #      var, var - self.memory[1, 2], var - self.memory[2, 2], var - self.memory[3, 2],
        #      var - self.memory[4, 2], var - self.memory[5, 2], var - self.memory[6, 2], var - self.memory[7, 2],
        #      var - self.memory[8, 2], var - self.memory[9, 2], var - self.memory[10, 2], var - self.memory[11, 2],
        #      var - self.memory[12, 2], var - self.memory[13, 2], var - self.memory[14, 2], var - self.memory[15, 2],
        #      var - self.memory[16, 2], var - self.memory[17, 2], var - self.memory[18, 2], var - self.memory[19, 2],
        #      ent, ent - self.memory[1, 3], ent - self.memory[2, 3], ent - self.memory[3, 3],
        #      ent - self.memory[4, 3], ent - self.memory[5, 3], ent - self.memory[6, 3], ent - self.memory[7, 3],
        #      ent - self.memory[8, 3], ent - self.memory[9, 3], ent - self.memory[10, 3], ent - self.memory[11, 3],
        #      ent - self.memory[12, 3], ent - self.memory[13, 3], ent - self.memory[14, 3], ent - self.memory[15, 3],
        #      ent - self.memory[16, 3], ent - self.memory[17, 3], ent - self.memory[18, 3], ent - self.memory[19, 3],
        #      mag], dim=0)

        # self.state = torch.stack(
        #     [wd - self.memory[1, 0], wd - self.memory[2, 0], wd - self.memory[3, 0], wd - self.memory[4, 0],
        #      wd - self.memory[5, 0], wd - self.memory[6, 0], wd - self.memory[7, 0], wd - self.memory[8, 0],
        #      wd - self.memory[9, 0], wd - self.memory[10, 0], wd - self.memory[11, 0], wd - self.memory[12, 0],
        #      wd - self.memory[13, 0], wd - self.memory[14, 0], wd - self.memory[15, 0], wd - self.memory[16, 0],
        #      wd - self.memory[17, 0], wd - self.memory[18, 0], wd - self.memory[19, 0],
        #      # score, score - self.memory[1, 1], score - self.memory[2, 1], score - self.memory[3, 1],
        #      # score - self.memory[4, 1], score - self.memory[5, 1], score - self.memory[6, 1],
        #      # score - self.memory[7, 1], score - self.memory[8, 1], score - self.memory[9, 1],
        #      # score - self.memory[10, 1], score - self.memory[11, 1], score - self.memory[12, 1],
        #      # score - self.memory[13, 1], score - self.memory[14, 1], score - self.memory[15, 1],
        #      # score - self.memory[16, 1], score - self.memory[17, 1], score - self.memory[18, 1],
        #      # score - self.memory[19, 1],
        #      var, var - self.memory[1, 2], var - self.memory[2, 2], var - self.memory[3, 2],
        #      var - self.memory[4, 2], var - self.memory[5, 2], var - self.memory[6, 2], var - self.memory[7, 2],
        #      var - self.memory[8, 2], var - self.memory[9, 2], var - self.memory[10, 2], var - self.memory[11, 2],
        #      var - self.memory[12, 2], var - self.memory[13, 2], var - self.memory[14, 2], var - self.memory[15, 2],
        #      var - self.memory[16, 2], var - self.memory[17, 2], var - self.memory[18, 2], var - self.memory[19, 2],
        #      ent, ent - self.memory[1, 3], ent - self.memory[2, 3], ent - self.memory[3, 3],
        #      ent - self.memory[4, 3], ent - self.memory[5, 3], ent - self.memory[6, 3], ent - self.memory[7, 3],
        #      ent - self.memory[8, 3], ent - self.memory[9, 3], ent - self.memory[10, 3], ent - self.memory[11, 3],
        #      ent - self.memory[12, 3], ent - self.memory[13, 3], ent - self.memory[14, 3], ent - self.memory[15, 3],
        #      ent - self.memory[16, 3], ent - self.memory[17, 3], ent - self.memory[18, 3], ent - self.memory[19, 3],
        #      mag], dim=0)

        self.state = torch.stack(
            [wd - self.memory[1, 0], wd - self.memory[2, 0], wd - self.memory[3, 0], wd - self.memory[4, 0],
             wd - self.memory[5, 0], wd - self.memory[6, 0], wd - self.memory[7, 0], wd - self.memory[8, 0],
             wd - self.memory[9, 0], wd - self.memory[10, 0], wd - self.memory[11, 0], wd - self.memory[12, 0],
             wd - self.memory[13, 0], wd - self.memory[14, 0], wd - self.memory[15, 0], wd - self.memory[16, 0],
             wd - self.memory[17, 0], wd - self.memory[18, 0], wd - self.memory[19, 0],
             score, score - self.memory[1, 1], score - self.memory[2, 1], score - self.memory[3, 1],
             score - self.memory[4, 1], score - self.memory[5, 1], score - self.memory[6, 1],
             score - self.memory[7, 1], score - self.memory[8, 1], score - self.memory[9, 1],
             score - self.memory[10, 1], score - self.memory[11, 1], score - self.memory[12, 1],
             score - self.memory[13, 1], score - self.memory[14, 1], score - self.memory[15, 1],
             score - self.memory[16, 1], score - self.memory[17, 1], score - self.memory[18, 1],
             score - self.memory[19, 1],
             # var, var - self.memory[1, 2], var - self.memory[2, 2], var - self.memory[3, 2],
             # var - self.memory[4, 2], var - self.memory[5, 2], var - self.memory[6, 2], var - self.memory[7, 2],
             # var - self.memory[8, 2], var - self.memory[9, 2], var - self.memory[10, 2], var - self.memory[11, 2],
             # var - self.memory[12, 2], var - self.memory[13, 2], var - self.memory[14, 2], var - self.memory[15, 2],
             # var - self.memory[16, 2], var - self.memory[17, 2], var - self.memory[18, 2], var - self.memory[19, 2],
             # ent, ent - self.memory[1, 3], ent - self.memory[2, 3], ent - self.memory[3, 3],
             # ent - self.memory[4, 3], ent - self.memory[5, 3], ent - self.memory[6, 3], ent - self.memory[7, 3],
             # ent - self.memory[8, 3], ent - self.memory[9, 3], ent - self.memory[10, 3], ent - self.memory[11, 3],
             # ent - self.memory[12, 3], ent - self.memory[13, 3], ent - self.memory[14, 3], ent - self.memory[15, 3],
             # ent - self.memory[16, 3], ent - self.memory[17, 3], ent - self.memory[18, 3], ent - self.memory[19, 3],
             mag], dim=0)


    def terminal_condition(self):
        if self.state[self.frame_num-1].item() >= 0.9:
            self.done = torch.tensor([1], device=self.device, dtype=torch.float)
        else:
            if self.max_wd != self.memory[torch.argmax(self.memory[:, 1]), 0].item():
                self.max_wd = self.memory[torch.argmax(self.memory[:, 1]), 0].item()
                self.check = 0
            elif self.max_wd == self.memory[torch.argmax(self.memory[:, 1]), 0].item():
                self.check += 1

            if self.check == 5:
                self.done = torch.tensor([1], device=self.device, dtype=torch.float)
            elif self.iteration >= self.iter_max:
                self.done = torch.tensor([1], device=self.device, dtype=torch.float)
            else:
                self.done = torch.tensor([0], device=self.device, dtype=torch.float)

    def set_action_scale(self, mag):
        self.scale *= np.round((-math.tanh(mag.item()*100-9)+1)/2*(1.0-0.03)+0.03, 2)
        self.max_action = np.round(self.max_action*self.scale, 3)


