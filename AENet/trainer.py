from __future__ import division, absolute_import
import data_loader
import model
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tensorboardX import SummaryWriter

'''dataset preparation'''
path = '/home/leewj/projects/projects/COXEM/EvaluationNetwork/dataset'
trainloader = data_loader.get_loader(dataset_path=path, num_classes=10, phase='train', shuffle=True, batch_size=90, num_workers=2)
valloader = data_loader.get_loader(dataset_path=path, num_classes=10, phase='val', shuffle=True, batch_size=90, num_workers=2)


'''utils'''
def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def get_clear_tensorboard():
    file_list = os.listdir("log_tensorboard")
    if file_list != None:
        for file in file_list:
            os.remove("log_tensorboard/%s" % (file))

ex_img, mag, ex_target = next(iter(trainloader))
ex_img = torchvision.utils.make_grid(ex_img)
imshow(ex_img, title=[ex_target])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model.EvalNet(), device_ids=[0, 1, 2], output_device=0)
model.to(device)
print(model)
print("Training model is loaded.")

optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-12, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False)
criterion = nn.MSELoss()


###################################################################################

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    accuracy = 0.0
    for batch_idx, (data, mag, target) in enumerate(train_loader):
        img, mag, target = data.to(device), mag.to(device), target.to(device).float().squeeze()
        data = torch.stack((img, mag), dim=1).squeeze()
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        for i in range(0, len(target)):
            if np.abs(output[i].item() - target[i].item()) <= 0.5:
                accuracy = accuracy + 1.0

    accuracy = 100. * accuracy / len(train_loader.dataset)
    train_loss_per_epoch.append(train_loss)
    train_acc_per_epoch.append(accuracy)
    return train_loss, train_loss_per_epoch, train_acc_per_epoch, accuracy

def evaluate(model, val_loader):
    model.eval()
    eval_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for data, mag, target in val_loader:
            img, mag, target = data.to(device), mag.to(device), target.to(device).float().squeeze()
            data = torch.stack((img, mag), dim=1).squeeze()
            output = model(data).squeeze()
            eval_loss += criterion(output, target).item()

            for i in range(0, len(target)):
                if np.abs(output[i].item() - target[i].item()) <= 0.5:
                    accuracy = accuracy + 1.0

    accuracy = 100. * accuracy / len(val_loader.dataset)
    eval_loss_per_epoch.append(eval_loss)
    eval_acc_per_epoch.append(accuracy)
    return eval_loss, eval_loss_per_epoch, eval_acc_per_epoch, accuracy

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch' : epoch,
        'State_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(state, filename)

print('Learning started.')

train_loss_per_epoch,  train_acc_per_epoch= [], []
eval_loss_per_epoch, eval_acc_per_epoch = [], []

model_name = '201201,MAG,240x320'
saved_model_name = './output/' + model_name + '.pth'
get_clear_tensorboard()
writer = SummaryWriter(logdir="log_tensorboard")

checking = 0

for epoch in range(1, 250 + 1):
    tic = time.time()
    train_loss, train_loss_per_epoch, train_acc_per_epoch, train_accuracy = train(model, trainloader, optimizer)
    eval_loss, eval_loss_per_epoch, eval_acc_per_epoch, eval_accuracy = evaluate(model, valloader)
    scheduler.step(eval_loss)

    writer.add_scalar('Train_Loss', train_loss, epoch)
    writer.add_scalar('Eval_Loss', eval_loss, epoch)
    writer.add_scalar('Train_Accuracy', train_accuracy, epoch)
    writer.add_scalar('Eval_Accuracy', eval_accuracy, epoch)
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

    toc = time.time()

    print('[{}] Train Loss: {:.4f}, Eval Loss: {:.4f}, Train Accuracy: {:.4f}%, Eval Accuracy: {:.4f}%, time: {:.3f}s/epoch, lr: {:f}'
        .format(epoch, train_loss, eval_loss, train_accuracy, eval_accuracy, toc - tic, optimizer.param_groups[0]['lr']))

    if epoch >= 10:
        if train_loss_per_epoch[epoch-1] < train_loss_per_epoch[epoch-2]:
            save_checkpoint(epoch, model, optimizer, saved_model_name)
            print(['Trained model is saved into the "' + saved_model_name + '".'])

print('Learning is over.')

writer.close()

plt.subplot(1, 2, 1)
plt.plot(train_loss_per_epoch, 'r', label = 'Train loss per epoch')
plt.plot(eval_loss_per_epoch, 'b', label = 'Val loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_per_epoch, 'r', label='Train Accuracy per epoch')
plt.plot(eval_acc_per_epoch, 'b', label = 'Val Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('%')
plt.title('Model Accuracy')
plt.legend()

plt.show()

os.makedirs('./output/' + model_name + '[result]')
np.savetxt('./output/' + model_name + '[result]/' + 'train_loss_per_epoch.txt', train_loss_per_epoch, fmt = '%.4f', delimiter=',')
np.savetxt('./output/' + model_name + '[result]/' + 'eval_loss_per_epoch.txt', eval_loss_per_epoch, fmt = '%.4f', delimiter=',')
np.savetxt('./output/' + model_name + '[result]/' + 'train_acc_per_epoch.txt', train_acc_per_epoch, fmt = '%.4f', delimiter=',')
np.savetxt('./output/' + model_name + '[result]/' + 'eval_acc_per_epoch.txt', eval_acc_per_epoch, fmt = '%.4f', delimiter=',')