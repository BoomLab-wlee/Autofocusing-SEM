import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch
from PIL import Image
import linecache
import torch.nn as nn
import model
import matplotlib.pyplot as plt
import time
import os

class ImagesFolder(data.Dataset):
    def __init__(self, root, num_classes, phase):
        """Initializes image paths and preprocessing module."""
        # Parameters setting
        assert phase in ["train", "val", "test"]
        self.num_classes = num_classes
        self.phase = phase
        self.phase_folder = phase

        # Path setting
        self.root = root
        self.image_paths = os.listdir(os.path.join(self.root, self.phase_folder))
        self.image_paths.sort()

        self.image_name = []
        self.data_paths = []

        self.check = []
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

        # Magnification Normalization
        mag = linecache.getline(txt_path, 10)
        mag = float(mag[4: len(mag) - 1]) / 10000

        if len(image_path) == 84:
            target_wd = float(image_path[len(image_path) - 35])
        else:
            target_wd = float(image_path[len(image_path) - 36])

        # target_wd = linecache.getline(txt_path, 37)
        # target_wd = float(target_wd[len(target_wd) - 2])

        # Data augmentation
        image = np.array(image).astype(np.float32) / 255.0

        if self.phase != "test":
            # Random brightness & contrast & gamma adjustment (linear and nonlinear intensity transform)
            # brightness_factor = random_factor(0.12)
            # contrast_factor = random_factor(0.20)
            # gamma_factor = random_factor(0.45)

            #
            # image = (brightness_factor - 1.0) + image
            # image = 0.5 + contrast_factor * (image - 0.5)
            # image = np.clip(image, 0.0, 1.0)
            # image = image ** gamma_factor
            # Image standard deviation normalization

            image = F.to_pil_image(image, mode="F")

            # Random horizontal flipping
            if random.random() < 0.5:
                image = F.hflip(image)

            # Random vertical flipping
            if random.random() < 0.5:
                image = F.vflip(image)

            image = np.array(image).astype(np.float32)
            image = image / np.std(image)

        else:
            image = image / np.std(image)

        # ToTensor
        transform = list()
        transform.append(T.ToTensor())  # ToTensor should be included before returning.
        transform = T.Compose(transform)

        if image.shape == (480, 640):
            crop1, crop2 = np.split(image, 2, axis=0)
            image11, image12 = np.split(crop1, 2, axis=1)
            image21, image22 = np.split(crop2, 2, axis=1)
            image = np.stack([image11, image12, image21, image22], axis = 2)
            mag = np.ones(24) * mag

        image = transform(image)
        return image, mag, target_wd, image_path

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


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

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

trainloader = get_loader(dataset_path = path, num_classes=10, phase='train', shuffle=True, batch_size=1, num_workers=0)
# valloader = get_loader(dataset_path = path, num_classes=10, phase='val', shuffle=True, batch_size=1, num_workers=0)
# testloader = get_loader(dataset_path = path, num_classes=10, phase='test', shuffle=True, batch_size=1, num_workers=0)

def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model = nn.DataParallel(AFNet(), device_ids=[0, 1])
test_model.to(device)
model_name = '200813,trained_model[Residual_noact_model240,L2reg,residual,MSE,lr=0.01,std,no_jitter,patience=7]'
checkpoint = torch.load('./output/' + model_name + '.pth')
test_model.load_state_dict(checkpoint['State_dict'])
relabelling = []

def tester(model, test_loader):
    model.eval()
    accuracy = 0.0
    with torch.no_grad():
        data, mag, target, check = next(iter(test_loader))
        # ex_img = torchvision.utils.make_grid(data)
        # imshow(ex_img)
        data, mag, target = data.to(device), mag.to(device), target.to(device).float().squeeze()
        output = model(data, mag).squeeze()
        output = torch.mean(output)

        if np.abs(output.item() - target.item()) <= 0.5:
            accuracy = accuracy + 1.0
        elif np.abs(output.item() - target.item()) >= 1.5:
            if target.item() >= 5:
                check = str(check)
                check = check[1:len(check)-2]
                relabelling.append([check, ' / ', str(target.item()), ' / ', str(output.item())])
                print(check, ' / ', str(target.item()), ' / ', str(output.item()))
            # img = Image.open(check)
            # new_path = check[0:len(check)-37] + str(int(np.round(output.item()))) + check[len(check)-36:len(check)-1]
            # img.save(new_path)
            # os.remove(check)
            # print('SUCCESS!!')

        pred_list.append(np.round(output.item(), 4))
        target_list.append(target.item())

    accuracy = 100. * accuracy
    eval_acc_per_epoch.append(accuracy)
    return accuracy, eval_acc_per_epoch, pred_list, target_list, relabelling

print('##################Evaluation is starting!##################')
eval_acc_per_epoch = []  # based on MSE
target_list = []
pred_list = []
elapsed_time = []
total_accuracy = 0.0
epoch = 150000
# epoch = 20

for epoch in range(1, epoch+1):
    accuracy, eval_acc_per_epoch, pred_list, target_list, relabelling = tester(test_model, trainloader)

    if epoch % 1000 == 0:
        np.savetxt('./output/' + model_name + '[result]/' + 'relabelling.txt', relabelling, fmt='%s', delimiter=',')
    print(epoch)

total_accuracy /= epoch

print("Target:", target_list)
print("Pred:", pred_list)

plt.figure(figsize=(8, 8))
plt.scatter(target_list, pred_list)
plt.xlabel('Target')
plt.ylabel('Pred')
plt.title('Target-Pred Result')
plt.grid(b = True, axis = 'both')
plt.xticks(np.arange(0, 10, step = 1))
plt.yticks(np.arange(0, 10, step = 1))

plt.show()

plt.subplot(2, 1, 1)
plt.scatter(np.arange(1, epoch + 1, step = 1), eval_acc_per_epoch, c = 'r')
plt.xlabel('Trial')
plt.ylabel('%')
plt.yticks(np.arange(50, 101, step = 10))
plt.title('Evaluation Accuracy per trial' + ' [Total Acc: ' + str(total_accuracy) + ']')

plt.subplot(2, 1, 2)
plt.plot(elapsed_time, 'r')
plt.xlabel('Trial')
plt.ylabel('Time[s]')
plt.title('Elapsed Time per trial')

plt.show()
print(relabelling)
np.savetxt('./output/' + model_name + '[result]/' + 'relabelling.txt', relabelling, fmt = '%s', delimiter=',')