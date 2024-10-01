import torch.nn as nn
import torch.nn.functional as F
import torch

'''
Receptive Field : 240 x 320
Result : 240 x 320 -> 1 x 1
Result : 480 x 640 -> 16 x 21

'''
class Evalnet(nn.Module):
    def __init__(self):
        super(Evalnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(3))
        self.layer2 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(32))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(64))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(64))
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer6 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(128))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(128))
        self.layer8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer9 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(256))
        self.layer10 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 4), stride=1, padding=0, bias=False), nn.BatchNorm2d(256))
        self.layer11 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer12 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(4, 5), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))
        self.layer13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(4, 5), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))
        self.layer14 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer15 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(4, 5), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))
        self.layer16 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(4, 5), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))
        self.layer17 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(4, 5), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))
        # Appending magnification
        self.layer18 = nn.Sequential(nn.Conv2d(513, 513, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(513))
        self.layer19 = nn.Sequential(nn.Conv2d(513, 513, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(513))
        self.layer20 = nn.Conv2d(513, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 7), stride=1, padding=0, bias=False), nn.BatchNorm2d(64))

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 7), stride=1, padding=0, bias=False), nn.BatchNorm2d(128))

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 7), stride=1, padding=0, bias=False), nn.BatchNorm2d(256))

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(7, 9), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))

        self.shortcut5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(7, 9), stride=1, padding=0, bias=False), nn.BatchNorm2d(512))

    def appending_magnification(self, out, mag):
        if out.size()[3] == 1:
            mag = torch.reshape(mag, [mag.size()[0], 1, 1, 1])
            out = torch.cat((out, mag.float()), 1)
        elif out.size()[3] == 21:
            mag = torch.reshape(mag, [mag.size()[0], 1, 16, 21])
            out = torch.cat((out, mag.float()), 1)
        else:
            raise RuntimeError("Please check the image and mag size.")
        return out

    def forward(self, data, mag):
        out = self.layer1(data)
        out = F.leaky_relu(self.layer2(out), 0.2)
        res = self.shortcut1(out)
        out = F.leaky_relu(self.layer3(out), 0.2)
        out = F.leaky_relu(self.layer4(out) + res, 0.2)
        out = self.layer5(out)
        res = self.shortcut2(out)
        out = F.leaky_relu(self.layer6(out), 0.2)
        out = F.leaky_relu(self.layer7(out) + res, 0.2)
        out = self.layer8(out)
        res = self.shortcut3(out)
        out = F.leaky_relu(self.layer9(out), 0.2)
        out = F.leaky_relu(self.layer10(out) + res, 0.2)
        out = self.layer11(out)
        res = self.shortcut4(out)
        out = F.leaky_relu(self.layer12(out), 0.2)
        out = F.leaky_relu(self.layer13(out) + res, 0.2)
        out = self.layer14(out)
        res = self.shortcut5(out)
        out = F.leaky_relu(self.layer15(out), 0.2)
        out = F.leaky_relu(self.layer16(out) + res, 0.2)
        out = F.leaky_relu(self.layer17(out), 0.2)
        out = self.appending_magnification(out, mag)
        out = F.leaky_relu(self.layer18(out), 0.2)
        out = F.leaky_relu(self.layer19(out), 0.2)
        out = self.layer20(out)
        out = F.relu6(out) * 9 / 6
        return out
