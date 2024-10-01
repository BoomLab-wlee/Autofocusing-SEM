import evaluator_model
import random
import time
import socket
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms as T
import torch.nn as nn
import torch
import os
import linecache
import shutil

# Socket initialization
sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
sc.settimeout(2)

# Connect
host = "127.0.0.1"
port = 9025

while True:
    print("Connecting...")
    res = sc.connect_ex((host, port))    
    if res == 0:
        print('ok')
        break
    else:
        print('fail - ', res)
        time.sleep(3)

# Send first
sGreeting = "Connected."
sent = sc.sendall(sGreeting.encode())
if sent is None:
    print(sGreeting)

time.sleep(0.005)
sStartComm = "COMM.START."
sent = sc.sendall(sStartComm.encode())
if sent is None:
    print(sStartComm)

time.sleep(0.005)
sReadyComm = "READY"
sc.sendall(sReadyComm.encode())
if sent is None:
    print(sReadyComm)

img_path = 'C:/aisem/ResponseImage.jpg'
par_path = 'C:/aisem/ResponseImage.txt'
save_path = 'C:/aisem/SEM_Connection_lee/data'

if os.path.isdir(save_path) != True:
    os.makedirs(save_path)

output_list = []

# Receiving loop
print('Data receiving loop from SEM is start.')
time.sleep(0.005)
trial = 0

while True:
    trial += 1
    
    if trial%3 == 1:
        sample = 'Grid'
    elif trial%3 == 2:
        sample = 'Tin'
    elif trial%3 == 0:
        sample = 'Au'
        
    d_save_path = save_path + '/' + sample + '/' + str(trial)
    if os.path.isdir(d_save_path) != True:
        os.makedirs(d_save_path)
        
    if sample == 'Grid':
        set_sem_position = [random.randrange(15000, 16500) / 1000, random.randrange(11500, 12000) / 1000]
        set_sem_mag = random.randrange(6000, 10000)
    elif sample == 'Tin':
        set_sem_position = [random.randrange(20000, 23000) / 1000, random.randrange(18600, 19400) / 1000]
        set_sem_mag = random.randrange(8000, 13000)
    elif sample == 'Au':
        set_sem_position = [random.randrange(15000, 16500) / 1000, random.randrange(26500, 28500) / 1000]
        set_sem_mag = random.randrange(9000, 15000)

    if set_sem_mag >= 10000:
        stride = 5
        init_wd = 19000
    else:
        stride = 50
        init_wd = 14000

    message = "POSITION" + ',' + str(set_sem_position[0]) + ',' + str(set_sem_position[1])
    sc.send(message.encode())
    time.sleep(5.0)

    for e in range(0, 250):
        wd = init_wd + stride*e
        message = "WDMAG" + "," + str(wd/1000) + "," + str(set_sem_mag)
        sc.send(message.encode())
        time.sleep(0.005)
        recv_data = sc.recv(1024)
        recv_data = recv_data.decode('euc-kr')
        print(recv_data)
        time.sleep(0.005)

        while True:
            if os.path.isfile(img_path) and os.path.isfile(par_path):
                shutil.move(img_path, d_save_path + '/' + str(wd) + '.jpg')
                shutil.move(par_path, d_save_path + '/' + str(wd) + '.txt')
                print("Image and Txt file is moved to saving directory.")
                time.sleep(0.005)
                break
            else:
                print('Waiting data...')
                time.sleep(0.5)

    if trial == 18:
        break

sc.close()
