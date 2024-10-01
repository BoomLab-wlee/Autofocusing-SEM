import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch
from PIL import Image, ImageFilter, ImageEnhance
import linecache
import matplotlib.pyplot as plt
import random

class ImagesFolder(data.Dataset):
    def __init__(self, root, num_classes, phase):
        """Initializes image paths and preprocessing module."""
        # Parameters setting
        assert phase in ["train", "val", "test", "test_red"]
        self.num_classes = num_classes
        self.phase = phase
        self.phase_folder = phase
        # Path setting
        self.root = root
        self.image_paths = os.listdir(os.path.join(self.root, self.phase_folder))
        self.image_paths.sort()

        self.image_name = []
        self.data_paths = []
        for i in range(len(self.image_paths)):
            data_path = os.path.join(self.root, self.phase_folder, self.image_paths[i])
            self.image_name = os.listdir(data_path)
            self.image_name = [file for file in self.image_name if file.endswith(".jpg")]
            for j in range(len(self.image_name)):
                self.data_paths.append(os.path.join(data_path, self.image_name[j]))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        # Random factor extraction
        def random_factor(factor):
            assert factor > 0
            return_factor = factor * np.random.randn() + 1
            if return_factor < 1 - factor:
                return_factor = 1 - factor
            if return_factor > 1 + factor:
                return_factor = 1 + factor
            return return_factor

        # Load image and mask files
        image_path = self.data_paths[index]  # Random index
        txt_path = image_path[0:len(image_path) - 4] + '.txt'
        image = Image.open(image_path)

        # Image channel split
        image = image.getchannel(0)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        if self.phase in ["train", "val", "test"]:
            mag = linecache.getline(txt_path, 10)
            mag = int(mag[4: len(mag) - 1]) / 100000
            target_wd = linecache.getline(txt_path, 37)
            target_wd = int(target_wd[len(target_wd) - 2])
        elif self.phase == "test_red":
            mag = linecache.getline(txt_path, 9)
            mag = int(mag[4: len(mag) - 1]) / 100000
            target_wd = int(image_path[len(image_path)-8:len(image_path)-4])


        # Data augmentation
        image = np.array(image).astype(np.float32) / 255.0

        if self.phase != "test":
            image = F.to_pil_image(image, mode="F")
            # Random horizontal flipping
            if random.random() < 0.5:
                image = F.hflip(image)

            # Random vertical flipping
            if random.random() < 0.5:
                image = F.vflip(image)

            image = np.array(image).astype(np.float32)
            image = (image - np.mean(image)) / np.std(image)
        else:
            image = (image - np.mean(image)) / np.std(image)

        mag = np.ones([image.shape[0], image.shape[1]]) * mag
        mag = np.array(mag).astype(np.float32)

        transform = T.Compose([T.ToPILImage(), T.RandomCrop(size=[240, 320]), T.ToTensor()])
        image, mag = transform(image), transform(mag)

        return image, mag, target_wd

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)

    def imshow(self, image, title=None):
        image = np.array(image)/255
        image = np.clip(image, 0, 1)
        plt.imshow(image, cmap='gray')
        if title is not None:
            plt.title(title)
        plt.pause(0.001)


def get_loader(dataset_path, num_classes, phase="train", shuffle=True, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""

    dataset = ImagesFolder(root=dataset_path,
                          num_classes=num_classes,
                          phase=phase)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader

def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)