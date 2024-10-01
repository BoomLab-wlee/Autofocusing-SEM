import time
from PIL import Image, ImageFilter
import numpy as np
import linecache
import torch
import torch.nn as nn
from torchvision import transforms as T
import os
import shutil
import evaluator_model

class Simulation_Client():
    def __init__(self):
        self.sample = ''
        self.idx = 0
        self.device = torch.device('cuda:0')
        self.data_path = ''
        self.data_length = ''
        self.target = 0
        self.random_noise = []

    def send(self, command, wd):
        if command == 'reset':
            self.sample = str(wd) + '_.txt'
            print('Off-line reinforcement learning data reset.')
            self.data_path = os.path.join(os.getcwd(), 'data', self.sample)
            d = open(self.data_path, 'r')
            self.data_length = len(d.readlines())-1
            d.close()

            infor = linecache.getline(self.data_path, 1)
            infor = infor.split(' ')

            linecache.clearcache()

            if infor != ['']:
                self.target = float(infor[3])/1000

        elif command =='WD':
            wd = np.round(wd*1000)
            self.idx = int(wd)+1
            time.sleep(0.005)

        elif command == 'noise':
            self.random_noise = np.random.normal(0, wd, size=self.data_length)

    def receive(self):
        while True:
            if os.path.exists(self.data_path):
                time.sleep(0.005)
                break
            else:
                print('Please check the data path...')
                time.sleep(0.005)
        infor = linecache.getline(self.data_path, self.idx)
        infor = infor.split(' ')

        linecache.clearcache()

        if infor != ['']:
            score, var, ent, mag = infor[6], infor[9], infor[12], infor[15]
            score, var, ent, mag = float(score), float(var), float(ent), float(mag)

            score = torch.tensor(score).to(self.device) + self.random_noise[self.idx-2]
            var = torch.tensor(var).to(self.device) + self.random_noise[self.idx-2]/100
            ent = torch.tensor(ent).to(self.device) + self.random_noise[self.idx-2]/10
            mag = torch.tensor(mag).to(self.device)/100000
            done = 0

            # time.sleep(0.005)

        elif infor == ['']:
            score, var, ent, mag = 0.0, 0.0, 0.0, 0.0
            score = torch.tensor(score).to(self.device)
            var = torch.tensor(var).to(self.device)
            ent = torch.tensor(ent).to(self.device)
            mag = torch.tensor(mag).to(self.device)
            done = 1
            print(self.sample)
            print('Out of possible WD range.')

        return score, var, ent, mag, done