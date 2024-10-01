import numpy as np
import torch
import environment
import utils
import agent
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tester():
    def __init__(self):
        self.hp = utils.Hyperparameters()
        self.env = environment.env()
        self.state_size, self.action_dim = self.env.action_space()
        self.replay_buffer = utils.ReplayBuffer(self.action_dim, self.hp.buffer_size, self.hp.batch_size)
        self.agent = agent.Agent(env=self.env, state_size=self.state_size, action_dim=self.action_dim, replay_buffer=self.replay_buffer)

        self.agent.load()
        self.retry = 0

    def test(self, mode, trial, init_wd, sample, sem_position=[0,0], sem_mag=0):
        result_history = []
        for i_e in range(1, trial + 1):
            while True:
                state, max_action = self.env.reset(mode=mode, init_Wd=init_wd, sample=sample, retry=self.retry, sem_position=sem_position, sem_mag=sem_mag)
                self.agent.set_max_action(max_action)

                wd_history, score_history = [], []
                iteration = 0

                score_history.append(np.round(state[19].item()*10, 2))
                tic = time.time()

                while True:
                    action = self.agent.select_action(state, noise=0)
                    next_state, target_action, done, wd, self.max_wd, self.max_score = self.env.step(action)
                    state = next_state

                    if iteration == 0:
                        wd_history.append(wd*1000 - np.round(action.item()*1000))

                    wd_history.append(wd * 1000)
                    score_history.append(np.round(state[19].item()*10, 2))
                    iteration += 1

                    if done:
                        result_history.append([self.max_wd, self.max_score, state[len(state)-1].item(), iteration, time.time()-tic])
                        if score_history[len(score_history)-1] < 3.0:
                            self.retry += 1
                        else:
                            self.retry = 0
                        break

                if self.retry == 0 or self.retry == 2:
                    break

            print('----    wd_history : ' + str(wd_history))
            print('----    score_history : ' + str(score_history))
            print('Trial: {:d}, Done, Elpased Time: {:.3f} sec'.format(i_e ,time.time()-tic))

        self.env.finish()
        return result_history
