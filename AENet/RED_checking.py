import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter, ImageEnhance
import linecache
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import model
import time
import os
import torchvision
from scipy.stats.stats import pearsonr

class ImagesLoader(data.Dataset):
    def __init__(self, root, phase):
        # Parameters setting
        self.phase = phase
        self.phase_folder = phase

        # Path setting
        self.root = root
        self.image_paths = os.listdir(os.path.join(self.root, self.phase_folder))
        # self.image_paths.sort(key=lambda name: int(name[0:len(name) - 4]))
        self.image_paths = [file for file in self.image_paths if file.endswith(".jpg")]

    def get_image(self, index):
        # Load image and mask files
        image_path = os.path.join(self.root, self.phase_folder, self.image_paths[index])
        txt_path = image_path[0:len(image_path) - 4] + '.txt'
        image = Image.open(image_path)

        # Image channel split
        image = image.getchannel(0)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        mag = linecache.getline(txt_path, 9)
        # mag = linecache.getline(txt_path, 10)
        mag = int(mag[4: len(mag) - 1]) / 100000
        # wd = int(image_path[len(image_path) - 9:len(image_path) - 4])
        wd = linecache.getline(txt_path, 21)
        wd = float(wd[3:len(wd)-1])

        # Data augmentation
        image = np.array(image).astype(np.float32) / 255.0

        var = np.std(image)/np.mean(image)
        image = (image - np.mean(image)) / np.std(image)

        marg = np.histogramdd(np.ravel(image), bins=256)[0] / image.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))


        # ToTensor
        mag = np.ones([image.shape[0], image.shape[1]]) * mag
        mag = np.array(mag).astype(np.float32)

        transform = list()
        transform.append(T.ToTensor())  # ToTensor should be included before returning.
        transform = T.Compose(transform)

        transform = T.Compose([T.ToTensor()])
        image, mag = transform(image), transform(mag)

        return image, mag, wd, var, entropy

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)

    def imshow(self, image, title=None):
        image = np.array(image)/255
        image = np.clip(image, 0, 1)
        plt.imshow(image, cmap='gray')
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def random_factor(self, factor):
        assert factor > 0
        return_factor = factor * np.random.randn() + 1
        if return_factor < 1 - factor:
            return_factor = 1 - factor
        if return_factor > 1 + factor:
            return_factor = 1 + factor
        return np.round(return_factor, 2)

data_num = np.arange(1, 92, 1)
phase = []

for i in range(0, len(data_num)):
    phase.append(str(data_num[i]))

phase = ['Butterx200', 'Butterx1000', 'Fabricx200', 'Fabricx500']

path = '/home/leewj/projects/projects/[2020][COXEM]SEM_Autofocusing/AENet/dataset/result_figure'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
test_model = nn.DataParallel(model.EvalNet(), device_ids=[1])
test_model.to(device)

model_name = '201201,MAG,240x320'
# model_name = '201130,MAG,240x320'

checkpoint = torch.load('./output/' + model_name + '.pth', map_location='cuda:1')
test_model.load_state_dict(checkpoint['State_dict'])

def checker(model, loader):
    model.eval()
    out_list = []
    wd_list = []
    var_list = []
    ent_list = []
    for index in range(0, loader.__len__()):
        with torch.no_grad():
            img, mag, wd, var, entropy = loader.get_image(index)

            img, mag = img.to(device), mag.to(device)
            data = torch.stack((img, mag), dim=1).squeeze()
            data = torch.reshape(data, [1, 2, 240, 320])

            output = model(data).squeeze()
            output = torch.median(output)

            out_list.append(np.round(output.item(), 4))
            wd_list.append(wd/1000)
            var_list.append(var)
            ent_list.append(entropy)

    return out_list, wd_list, var_list, ent_list

for p in phase:
    loader = ImagesLoader(root=path, phase=p)
    out_list, wd_list, var_list, ent_list = checker(test_model, loader)
    print(out_list)
    print(wd_list)

    plt.plot(wd_list, out_list, '.')
    plt.xlabel('WD')
    plt.ylabel('Out')
    plt.title(p + ': WD - score')
    plt.grid(b = True, axis = 'both')
    plt.yticks(np.arange(0, 10, step = 1))
    plt.show()

    save_path = "//home/leewj/projects/projects/[2020][COXEM]SEM_Autofocusing/AENet/v2_normal_image_with_mag_image/score_result/" + str(p)
    np.savetxt(save_path + "_out.txt", out_list, fmt='%.4f', delimiter=',')
    np.savetxt(save_path + "_wd.txt", wd_list, fmt='%.3f', delimiter=',')
    np.savetxt(save_path + "_var.txt", var_list, fmt='%.3f', delimiter=',')
    np.savetxt(save_path + "_ent.txt", ent_list, fmt='%.3f', delimiter=',')