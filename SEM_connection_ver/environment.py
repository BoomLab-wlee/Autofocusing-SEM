import socket_client
import random
import torch
import numpy as np
import time
import math

class env():
    def __init__(self):
        self.device = torch.device('cuda')
        self.client = socket_client.Socket_Client()
        self.mode = ''

        self.frame_num= 20
        self.state = []
        self.state_size = (self.frame_num - 1) + (self.frame_num * 3) + 1
        self.action_dim = 1
        self.max_action = 1.2
        self.next_wd = 0
        self.reward = 0
        self.done = 0
        self.iteration = 0
        self.iter_max = 20
        self.scale = 1
        self.memory = []
        self.check = 0
        self.sequence = ''
        self.max_wd = 0
        self.max_score = 0
        self.target = 0

        self.sample = ''
        self.init_wd = 0.0

    def action_space(self):
        '''Define SEM client action dimension'''
        return self.state_size, self.action_dim

    def reset(self, mode, init_wd, sample, retry, sem_position, sem_mag):
        self.sample = sample

        if retry == 0:
            self.init_wd = init_wd
            # Randomly selects withing the magnification, x-y position of the stage set according to the type of sample.
            if mode == 'AutoTest':
                if self.sample == 'Grid':
                    self.set_sem_position = [random.randrange(15000, 16500) / 1000, random.randrange(11500, 12000) / 1000]
                    self.set_sem_mag = random.randrange(200, 7000)
                elif self.sample == 'Tin':
                    self.set_sem_position = [random.randrange(20000, 23000) / 1000, random.randrange(18600, 19400) / 1000]
                    self.set_sem_mag = random.randrange(200, 10000)
                elif self.sample == 'Au':
                    self.set_sem_position = [random.randrange(15000, 16500) / 1000, random.randrange(26500, 28500) / 1000]
                    self.set_sem_mag = random.randrange(10000, 30000)
                else:
                    assert("Please set the sample. (For example, sample = 'Tin')")

            elif mode == 'ManualTest':
                self.set_sem_position = sem_position
                self.set_sem_mag = sem_mag

            if self.set_sem_mag > 8500:
                self.init_wd =+ 1.0
            elif self.set_sem_mag > 10000:
                    self.init_wd =+ 2.0

            self.client.sem_send('WDMAG', [self.init_wd, self.set_sem_mag])
            time.sleep(0.005)
            self.client.sem_send('POSITION', self.set_sem_position)
            time.sleep(0.005)
            print('SEM is randomly initialized')

        elif retry == 1:
            self.init_wd = self.max_wd
            self.client.sem_send('WDMAG', [self.init_wd, self.set_sem_mag])
            time.sleep(0.005)
            print('SEM auto-focusing try again')


        wd, score, var, ent, mag = self.client.sem_receive()
        time.sleep(0.05)

        self.max_action = 1.2
        self.scale = 1
        self.memory = torch.zeros([self.frame_num, 5], device=self.device)
        self.state = torch.zeros([1, self.state_size], device=self.device)

        self.set_action_scale()

        self.iteration = 0
        self.check = 0

        self.initial_searching()

        return self.state, self.max_action

    def set_action_scale(self):
        self.scale *= np.round((-math.tanh(self.set_sem_mag/1000-9)+1)/2*(1.0-0.03)+0.03, 2)
        self.max_action = np.round(self.max_action*self.scale, 3)
        
    def initial_searching(self):
        stride = np.round(0.4*self.scale, 3)

        if self.sequence == 'Forward':
            stride = stride
        elif self.sequence == 'Inverse':
            stride = -stride

        wd2, score2, var2, ent2, mag2 = torch.zeros([5], device=self.device)
        wd3, score3, var3, ent3, mag3 = torch.zeros([5], device=self.device)

        while True:
            self.iteration += 1
            self.next_wd = self.init_wd + stride*self.iteration
            self.client.sem_send('WDMAG', [self.next_wd, self.set_sem_mag])
            time.sleep(0.005)
            wd, score, var, ent, mag = self.client.sem_receive()

            print('Init Processing : Score: {:.2f}'
              .format(np.round(score.item(), 2)))

            if score >= 1.8 and self.iteration >= 4:
                print('Initial_processing is done.')
                self.iteration = 0
                break
            elif self.next_wd >= 50.0:
                print('Initial_processing is Failed, retry.')
                stride = stride*0.5
                self.iteration = 0
            else:
                wd3, score3, var3, ent3, mag3 = wd2, score2, var2, ent2, mag2
                wd2, score2, var2, ent2, mag2 = wd, score, var, ent, mag

        wd3, score3, var3, ent3, mag3 = self.state_scaling(wd3, score3, var3, ent3, mag3)
        wd2, score2, var2, ent2, mag2 = self.state_scaling(wd2, score2, var2, ent2, mag2)

        self.state_memory(wd3, score3, var3, ent3, mag3)
        self.state_memory(wd2, score2, var2, ent2, mag2)
        self.set_state(wd, score, var, ent, mag)

    def step(self, action):
        self.iteration += 1

        action = action.item()
        self.next_wd = np.round(action+self.next_wd, 3)

        self.client.sem_send('WDMAG', [self.next_wd, self.set_sem_mag])
        time.sleep(0.005)
        wd, score, var, ent, mag = self.client.sem_receive()
        self.set_state(wd, score, var, ent, mag)
        
        self.get_optimal_action(action)
        self.terminal_condition()

        print('[Sample= ' + self.sample + ' Mag= {}] WD: {:.3f}, Score: {:.2f}, Action: {:.3f}'
              .format(self.set_sem_mag, np.round(self.next_wd, 3),
                      np.round(self.state[self.frame_num - 1].item() * 10, 2), np.round(action, 3)))

        if self.done == 1:
            self.max_wd = self.memory[torch.argmax(self.memory[:, 1]), 0].item()*self.max_action
            self.max_score = self.memory[torch.argmax(self.memory[:, 1]), 1].item()*10
            self.client.sem_send('WDMAG', [self.max_wd, self.set_sem_mag])
            time.sleep(5.0)

        return self.state, self.target_action, self.done, self.next_wd, self.max_wd, self.max_score

    def state_scaling(self, wd, score, var, ent, mag):
        wd = wd/self.max_action
        score = score/10
        var = var
        ent = ent/100
        mag = mag
        return wd, score, var, ent, mag

    def state_memory(self, wd, score, var, ent, mag):
        if torch.sum(self.memory).item() == 0:
            for i in range(0, self.frame_num - 1):
                self.memory[i, :] = torch.stack([wd, score, var, ent, mag])

        for i in range(0, self.frame_num - 1):
            i = -i + (self.frame_num - 2)
            self.memory[i+1, :] = self.memory[i, :]
        self.memory[0, :] = torch.stack([wd, score, var, ent, mag])

    def set_state(self, wd, score, var, ent, mag):
        wd, score, var, ent, mag = self.state_scaling(wd, score, var, ent, mag)
        self.state_memory(wd, score, var, ent, mag)

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
             var, var - self.memory[1, 2], var - self.memory[2, 2], var - self.memory[3, 2],
             var - self.memory[4, 2], var - self.memory[5, 2], var - self.memory[6, 2], var - self.memory[7, 2],
             var - self.memory[8, 2], var - self.memory[9, 2], var - self.memory[10, 2], var - self.memory[11, 2],
             var - self.memory[12, 2], var - self.memory[13, 2], var - self.memory[14, 2], var - self.memory[15, 2],
             var - self.memory[16, 2], var - self.memory[17, 2], var - self.memory[18, 2], var - self.memory[19, 2],
             ent, ent - self.memory[1, 3], ent - self.memory[2, 3], ent - self.memory[3, 3],
             ent - self.memory[4, 3], ent - self.memory[5, 3], ent - self.memory[6, 3], ent - self.memory[7, 3],
             ent - self.memory[8, 3], ent - self.memory[9, 3], ent - self.memory[10, 3], ent - self.memory[11, 3],
             ent - self.memory[12, 3], ent - self.memory[13, 3], ent - self.memory[14, 3], ent - self.memory[15, 3],
             ent - self.memory[16, 3], ent - self.memory[17, 3], ent - self.memory[18, 3], ent - self.memory[19, 3],
             mag], dim=0)

    def terminal_condition(self):
        if self.state[self.frame_num - 1].item() >= 0.9:
            self.done = torch.tensor([1], device=self.device, dtype=torch.float)
        else:
            if self.max_wd != self.memory[torch.argmax(self.memory[:, 1]), 0].item():
                self.max_wd = self.memory[torch.argmax(self.memory[:, 1]), 0].item()
                self.max_score = self.memory[torch.argmax(self.memory[:, 1]), 1].item()*10
                self.check = 0
            elif self.max_wd == self.memory[torch.argmax(self.memory[:, 1]), 0].item():
                self.check += 1

            if self.check == 8:
                self.done = torch.tensor([1], device=self.device, dtype=torch.float)
            elif self.iteration >= self.iter_max:
                self.done = torch.tensor([1], device=self.device, dtype=torch.float)
            else:
                self.done = torch.tensor([0], device=self.device, dtype=torch.float)

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

    def finish(self):
        self.client.sem_disconnect()


