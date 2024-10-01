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
save_path = 'C:/aisem/SEM_Connection_lee\data/compare_data'

if os.path.isdir(save_path) != True:
    os.makedirs(save_path)

output_list = []

# Receiving loop
print('Data receiving loop from SEM is start.')
time.sleep(0.005)

    
sample = 'Butterfly'
set_sem_mag = 2000
WD = [4000, 11516, 11406]
name = sample + '_trial_X' + str(set_sem_mag)

d_save_path = save_path + '/' + sample + '/' + name
if os.path.isdir(d_save_path) != True:
    os.makedirs(d_save_path)

for e in range(0, 3):
    wd = WD[e]
    message = "WDMAG" + "," + str(wd/1000) + "," + str(set_sem_mag)
    sc.send(message.encode())
    time.sleep(1.0)
    recv_data = sc.recv(1024)
    recv_data = recv_data.decode('euc-kr')
    print(recv_data)
    time.sleep(1.0)

    while True:
        if os.path.isfile(img_path) and os.path.isfile(par_path):
            shutil.move(img_path, d_save_path + '/' + str(e) + '.jpg')
            shutil.move(par_path, d_save_path + '/' + str(e) + '.txt')
            print("Image and Txt file is moved to saving directory.")
            time.sleep(0.005)
            break
        else:
            print('Waiting data...')
            time.sleep(0.5)


sc.close()
