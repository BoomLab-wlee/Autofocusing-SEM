import torch
import torch.nn as nn
import model
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import data_loader
import torchvision
from scipy.stats.stats import pearsonr

'''dataset making'''

path = '/home/leewj/projects/projects/COXEM/EvaluationNetwork/dataset'
trainloader = data_loader.get_loader(dataset_path = path, num_classes=10, phase='train', shuffle=True, batch_size=1, num_workers=0)
valloader = data_loader.get_loader(dataset_path = path, num_classes=10, phase='val', shuffle=True, batch_size=1, num_workers=0)
testloader = data_loader.get_loader(dataset_path = path, num_classes=10, phase='test', shuffle=True, batch_size=1, num_workers=0)
rtestloader = data_loader.get_loader(dataset_path = path, num_classes=10, phase='test_red', shuffle=True, batch_size=1, num_workers=0)

'''dataset test'''
def imshow(inp, title = None):
    inp = inp.cpu().detach()
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model = nn.DataParallel(model.EvalNet(), device_ids=[0])
test_model.to(device)
model_name = '201201,MAG,240x320'
checkpoint = torch.load('./output/' + model_name + '.pth')
test_model.load_state_dict(checkpoint['State_dict'])

def tester(model, test_loader):
    model.eval()
    accuracy = 0

    with torch.no_grad():
        data, mag, target = next(iter(test_loader))

        img, mag, target = data.to(device), mag.to(device), target.to(device).float().squeeze()
        data = torch.stack((img, mag), dim=1).squeeze()
        data = torch.reshape(data, [1, 2, 240, 320])
        output = model(data).squeeze()
        output = torch.median(output)

        if np.abs(output.item() - target.item()) <= 0.5:
            accuracy = 1

        precision = 1 - 0.1 * np.abs(output.item() - target.item())

        pred_list.append(np.round(output.item(), 4))
        target_list.append(target.item())

    precision = 100. * precision
    eval_acc_per_epoch.append(accuracy)
    eval_pre_per_epoch.append(precision)
    return accuracy, eval_acc_per_epoch, pred_list, target_list, precision, eval_pre_per_epoch

print('##################Evaluation is starting!##################')
eval_acc_per_epoch = []
eval_pre_per_epoch = []
target_list = []
pred_list = []
elapsed_time = []
total_accuracy = 0.0
total_precision = 0.0
epoch = 150

for trial in range(1, epoch+1):
    tic = time.time()
    accuracy, eval_acc_per_epoch, pred_list, target_list, precision, eval_pre_per_epoch = tester(test_model, testloader)
    total_accuracy += accuracy
    total_precision += precision
    toc = time.time()
    elapsed_time.append(tic - toc)
    print('[{}] Eval Precision: {:.4f}%, Eval Accuracy: {:.1f}, time: {:.4f}s/trial, Target: {:.1f}, Output: {:.4f}'
          .format(trial, precision, accuracy, toc - tic, target_list[trial - 1], pred_list[trial - 1]))

total_accuracy /= (epoch / 100)
total_precision /= epoch

print("Target:", target_list)
print("Pred:", pred_list)
plt.plot(np.arange(0, 9.5, 0.5), pearsonr(target_list, pred_list)[0] * np.arange(0, 9.5, 0.5), 'r--')
plt.plot(target_list, pred_list, 'bo')
plt.xlabel('Target')
plt.ylabel('Pred')
plt.title('Target-Pred Result, R = ' + str(np.round(pearsonr(target_list, pred_list)[0], 3)))
plt.grid(b = True, axis = 'both')
plt.yticks(np.arange(0, 10, step = 1))
plt.xticks(np.arange(0, 10, step = 1))
plt.show()
