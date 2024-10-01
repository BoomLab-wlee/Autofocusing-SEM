import numpy as np
import torch
from collections import namedtuple, deque
import random
import os

"""
Buffer to store tuples of experience replay
"""

class ReplayBuffer(object):
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_dim, buffer_size, batch_size):
		"""Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
		self.action_dim = action_dim
		self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
		self.batch_size = batch_size

	def add(self, state, action, target_action, done):
		"""Add a new experience to memory."""
		self.memory.append((state, action, target_action, done))

	def sample(self):
		"""Randomly sample a batch of experiences from memory."""

		batch = random.sample(self.memory, self.batch_size)
		states, actions, target_action, dones = zip(*batch)
		states = torch.cat(states)
		actions = torch.cat(actions)
		target_actions = torch.cat(target_action)
		dones = torch.cat(dones)
		return (states, actions, target_actions, dones)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

class Hyperparameters(object):
	def __init__(self):
		self.episodes = 200  # episode 반복횟수
		self.batch_size = 100 # 배치 크기
		self.buffer_size = 1000

	def print_args(self):
		print('-------------------------')
		print('Hyper Parameter Settings')
		print('-------------------------')
		for var in vars(self):
			value = getattr(self, var)
			print(var + ': ' + str(value))
		print('-------------------------')
