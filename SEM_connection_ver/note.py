import socket
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

class Socket_Client():
    def __init__(self):
        self.sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.sc.settimeout(1)
        self.host = "127.0.0.1"
        self.port = 9025
        self.device = torch.device('cuda')
        self.img_path = 'C:/aisem/ResponseImage.jpg'
        self.par_path = 'C:/aisem/ResponseImage.txt'
        self.save_path = 'C:/aisem/SEM_Connection_lee/saved_data/' + str(time.strftime('%y%m%d%H%M', time.localtime(time.time())))

        self.message = ''
        self.transform = list()
        self.transform.append(T.ToTensor())
        self.transform = T.Compose(self.transform)

        self.load_eval_net()

        self.sem_connect()
        recv_data = self.sc.recv(1024)
        recv_data = recv_data.decode('euc-kr')

    def sem_connect(self):
        while True:
            print("Connecting...")
            res = self.sc.connect_ex((self.host, self.port))
            if res == 0:
                break
            else:
                print('fail - ', res)
                time.sleep(3)
        sGreeting = "Connected."
        sent = self.sc.sendall(sGreeting.encode())
        if sent is None:
            print(sGreeting)
        time.sleep(0.005)

        sStartComm = "COMM.START."
        sent = self.sc.sendall(sStartComm.encode())
        if sent is None:
            print(sStartComm)
        time.sleep(0.005)

        sReadyComm = "READY"
        sent = self.sc.sendall(sReadyComm.encode())
        if sent is None:
            print(sReadyComm)

    def sem_send(self, command, action):
        self.message = command + ',' + str(action[0]) + ',' + str(action[1])  # SEM control message
        print(self.message)
        time.sleep(0.005)
        self.sc.send(self.message.encode())
        time.sleep(0.005)
        
        if command == 'WDMAG':
            recv_data = self.sc.recv(1024)
            recv_data = recv_data.decode('euc-kr')
            # print(recv_data)
            time.sleep(0.5)
        elif command == 'POSITION':
            time.sleep(4.0)
            print('SEM x-y position is changed.')

    def sem_receive(self, saving=False):
        while True:
            if os.path.exists(self.img_path):
                img = Image.open(self.img_path)
                break
            else:
                print('Waiting img...')
                time.sleep(0.005)

        img = img.getchannel(0)
        # A median filter is applied to the image to reduce image noise.
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = np.array(img).astype(np.float32) / 255.0

        var = np.std(img) / np.mean(img)
        img = (img - np.mean(img)) / np.std(img)

        # image entropy
        marg = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))
        
        mag = linecache.getline(self.par_path, 9)
        wd = linecache.getline(self.par_path, 21)

        linecache.clearcache()
        
        mag = float(mag[4: len(mag) - 1]) / 100000.0
        wd = float(wd[3: len(wd) - 4]) / 1000.0

        mag_img = np.ones([img.shape[0], img.shape[1]]) * mag
        mag_img = np.array(mag_img).astype(np.float32)

        wd = torch.tensor(wd).to(self.device)
        var = torch.tensor(var).to(self.device)
        entropy = torch.tensor(entropy).to(self.device)
        mag = torch.tensor(mag).to(self.device)

        img, mag_img = self.transform(img), self.transform(mag_img)
        img.to(self.device)
        mag_img.to(self.device)


        with torch.no_grad():
            time.sleep(0.005)
            data = torch.stack((img, mag_img), dim=1).squeeze()
            data = torch.reshape(data, [1, 2, img.size()[1], img.size()[2]])
            score = self.eval_net(data).squeeze()
            score = torch.median(score)

        if saving == True:
            if os.path.isdir(self.save_path) != True:
                os.makedirs(self.save_path)
            shutil.move(self.img_path, self.save_path + '/' + str(time.strftime('%H%M%S', time.localtime(time.time()))) + '.jpg')
            shutil.move(self.par_path, self.save_path + '/' + str(time.strftime('%H%M%S', time.localtime(time.time()))) + '.txt')
            print("Image and Par file is moved to saving directory.")
            time.sleep(0.005)

        elif saving == False:
            os.remove(self.img_path)
            os.remove(self.par_path)
            print("Image and Par file is removed.")
            time.sleep(0.005)

        return wd, score, var, entropy, mag

    def sem_disconnect(self):
        sFinish = 'FINISH'
        bindata = bytes(sFinish, 'euc-kr')
        self.sc.send(bindata)
        self.sc.close()
        print('SME connection is finished.')

    def load_eval_net(self):
        try:
            self.eval_net = nn.DataParallel(evaluator_model.EvalNet(), device_ids=[0])
            self.eval_net.to(self.device)
            self.eval_net.eval()
            model_name = 'Trained_network/EvalNet_201201_MAG_240x320.pth'
            checkpoint = torch.load(model_name)
            self.eval_net.load_state_dict(checkpoint['State_dict'])

            print('Saved state dict of Evaluation Network is loaded.')
        except:
            raise RuntimeError("Can not load trained model.")


test = Socket_Client()
