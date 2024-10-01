import numpy as np
import torch
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import environment
import utils
import agent


RL_name = 're_train_actor_210107'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp = utils.Hyperparameters()
env = environment.env(mode='train')
state_size, action_dim = env.action_space()
replay_buffer = utils.ReplayBuffer(action_dim, hp.buffer_size, hp.batch_size)
agent = agent.Agent(env=env, state_size=state_size, action_dim=action_dim, replay_buffer=replay_buffer)

past_episode = 0
agent.load()

hp.print_args()

def observe(env, batch_size):
    """run episodes while taking random actions and filling replay_buffer"""

    time_steps = 0
    state, max_action = env.reset()
    agent.set_max_action(max_action)
    noise = 0.5

    while time_steps < batch_size:
        action = agent.select_action(state, noise)
        next_state, target_action, done, wd = env.step(action)
        
        action = target_action
        next_state, target_action, done, wd = env.step(action)

        agent.store_transition(state, action, target_action, done)

        state = next_state
        time_steps += 1

        if done:
            state, max_action = env.reset()
            agent.set_max_action(max_action)

        print("\rPopulating Buffer {}/{}.".format(time_steps, batch_size), end="")

def train(agent, env, total_episode, p_e=past_episode):
    """Train the agent for exploration steps"""
    train_loss_per_episode = []
    train_accuracy_per_episode = []

    for i_e in range(p_e, p_e + total_episode + 1):
        state, max_action = env.reset()
        agent.set_max_action(max_action)

        wd_history = []
        score_history = []
        target_history = []
        noise = 0.5 * (0.98 ** p_e)
        episode_timesteps = 0

        episode_loss, episode_accuracy = 0, 0

        while True:
            action = agent.select_action(state, noise)
            ext_state, target_action, done, wd = env.step(action)
            agent.store_transition(state, action, target_action, done)
            state = next_state
            wd_history.append(wd * 1000)
            score_history.append(np.round(state[19].item() * 10, 2))
            target_history.append(np.round(target_action.item()*1000))

            episode_timesteps += 1

            noise *= .9995
            loss, accuracy = agent.train(replay_buffer)
            episode_loss += loss.item()
            episode_accuracy += accuracy

            if done:
                break

            if i_e % 300 == 0:
                agent.save(i_e, RL_name)

        train_loss_per_episode.append(episode_loss / episode_timesteps)
        train_accuracy_per_episode.append(episode_accuracy / episode_timesteps)

        state = state.squeeze()

        print('\rEpisode {}\tLoss: {:.3f}\tAccurcay: {:.3f}\tStep: {:d}\tfinal_state: [{:.1f}, {:.2f}]'
              .format(i_e, loss.item(), accuracy, episode_timesteps, wd * 1000, np.round(state[19].item() * 10, 2)))
        print('----    wd_history : ' + str(wd_history))
        print('----    score_history : ' + str(score_history))
        print('----    target_history : ' + str(target_history))

    return train_loss_per_episode, train_accuracy_per_episode


# Populate replay buffer
observe(env, hp.batch_size)
train_loss_per_episode, train_accuracy_per_episode = train(agent, env, total_episode=hp.episodes)

agent.save(past_episode + hp.episodes, RL_name)

fig = plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.plot(np.arange(1, len(train_loss_per_episode)+1), train_loss_per_episode)
plt.ylabel('Loss')
plt.xlabel('Episode #')

plt.subplot(1,2,2)
plt.plot(np.arange(1, len(train_accuracy_per_episode)+1), train_accuracy_per_episode)
plt.ylabel('Accuracy')
plt.xlabel('Episode #')
fig.show()

env.finish()
