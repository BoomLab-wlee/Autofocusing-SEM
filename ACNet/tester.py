import numpy as np
import torch
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import environment
import utils
import agent
import scipy.io

# RL_name = 'result_210125(random_noise_and_init_WD_max_action)'
# RL_name = 'result_210122(random_noise_and_init_WD_max_action)'
RL_name = '210927_no_AENet'
RL_name = '210927_no_low_score'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp = utils.Hyperparameters()
env = environment.env(mode='test')
state_size, action_dim = env.action_space()
replay_buffer = utils.ReplayBuffer(action_dim, hp.buffer_size, hp.batch_size)
agent = agent.Agent(env=env, state_size=state_size, action_dim=action_dim, replay_buffer=replay_buffer)

past_episode = 0
past_episode = agent.load(RL_name)

hp.print_args()
total_wd_list = []
action_list = []

def test(agent, env, sample):
    """Train the agent for exploration steps"""

    for i_e in range(len(sample)):
        state, max_action = env.reset(sample=sample[i_e])
        agent.set_max_action(max_action)

        wd_history = []
        score_history = []
        noise = 0.5
        episode_timesteps = 0
        score_history.append(np.round(state[19].item() * 10, 2))
        action_history = []
        target_action_history = []

        while True:
            action = agent.select_action(state, noise)
            next_state, target_action, done, wd = env.step(action)
            state = next_state
            if wd_history == []:
                wd_history.append(wd * 1000 - np.round(action.item()*1000))
            if wd*1000 <= 0:
                wd = 1.0

            wd_history.append(wd * 1000)
            action_history.append(action.item()*1000)
            target_action_history.append(target_action.item()*1000)
            score_history.append(np.round(state[19].item()*10, 2))
            episode_timesteps += 1

            if done:
                break

        state = state.squeeze()
        print('\rTest# {}\tStep: {:d}\tfinal_state: [{:.1f}, {:.2f}]'
              .format(i_e, episode_timesteps, wd * 1000, np.round(state[19].item() * 10, 2)))
        print('----    wd_history : ' + str(wd_history))
        print('----    score_history : ' + str(score_history))
        total_wd_list.append(wd_history)
        action_list.append(action_history)
        action_list.append(target_action_history)


        fig = plt.figure(figsize=(8, 8))
        plt.plot(wd_history, score_history, '.-')
        plt.ylabel('Score')
        plt.xlabel('WD')
        plt.scatter(wd_history[0], score_history[0], c='g')
        plt.scatter(wd_history[len(wd_history) - 1], score_history[len(score_history) - 1], c='r')
        plt.title('Sample: ' + str(sample[i_e]))
        # plt.xlim([1000, 14000])
        plt.ylim([0.0, 9.5])
        # fig.show()
        # plt.close(fig)

sample = np.arange(1, 60, 1)

test(agent, env, sample)
print(total_wd_list)
np.savetxt('210927_no_low_score_result.txt', total_wd_list, fmt='%s', delimiter=',')
np.savetxt('210927_no_low_score_action_result.txt', action_list, fmt='%s', delimiter=',')
# scipy.io.savemat('SL_terminal_result.mat', total_wd_list)