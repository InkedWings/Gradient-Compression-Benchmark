'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd

import os
import argparse

from models import *
from optim_comp import CompSGD
from memory.memory_chooser import memory_chooser
from compression.compression_chooser import compression_chooser
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model_name', default='none', help='model name to be loaded or saved')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true',
                    help='test checkpoint')
#TODO Add SZ lossy compressor
parser.add_argument('--compression', default='none', help='topk, qsgd')
parser.add_argument('--memory', default='none', help='residual, dgc')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers simulated')
parser.add_argument('--compression_ratio', type=float, default=0.1,
                        help='default compression ratio for sparsification techniques')
parser.add_argument('--quantum_num', type=int, default=8,
                        help='number of quantisation levels used in QSGD')

args = parser.parse_args()

wdir = "/home/zye25/Gradient-Compression-Benchmark/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

loss_data = {}
loss_data['train'] = []
loss_data['val'] = []

acc_data = {}
acc_data['train'] = []
acc_data['val'] = []

x_epoch = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
assert args.model_name != 'none', 'Model Name is not valid!'
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    print("Using", torch.cuda.device_count(), "GPUs!")
    print("DataParallel!!!")
    cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{model_name}.pth'.format(model_name=args.model_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

compressor = compression_chooser(args)
memory = memory_chooser(args)
print("Using Compressor: ", str(compressor))
print("Using memory: ", str(memory))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
optimizer = CompSGD(optimizer, net.named_parameters(),args.num_workers, compressor, memory)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global loss_data
    global acc_data
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        worker_num = (batch_idx+1) % args.num_workers   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.compress_step(worker_num)
        if worker_num == 0:
            optimizer.assign_grads()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("epoch: %d train loss: %.3f train acc: %.3f" % (epoch, train_loss/len(trainloader), 100.*correct/total))
    loss_data['train'].append(train_loss/len(trainloader))
    acc_data['train'].append(100.*correct/total)


def test(epoch):
    global best_acc
    global loss_data
    global acc_data
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("epoch: %d val loss: %.3f val acc: %.3f" % (epoch, test_loss/len(testloader), 100.*correct/total))
    loss_data['val'].append(test_loss/len(testloader))
    acc_data['val'].append(100.*correct/total)       

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{model_name}.pth'.format(model_name=args.model_name))
        best_acc = acc

# Data Visualization
def fig_plot(x_epoch, loss_data, acc_data):
    fig_loss = plt.figure()
    plt.title(args.model_name + " loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_epoch, loss_data['train'], color='lightblue',linestyle='solid', marker='.', label='train')
    plt.plot(x_epoch, loss_data['val'], color='orange',linestyle='solid', marker='.', label='val')
    plt.legend()
    fig_loss.savefig(wdir+"out_fig/loss/{model_name}.jpg".format(model_name=args.model_name))

    fig_acc = plt.figure()
    plt.suptitle(args.model_name + " accuracy")
    plt.xlabel("epoch")
    plt.ylabel("top1 accuracy")
    plt.plot(x_epoch, acc_data['train'], color='lightblue',linestyle='solid', marker='.', label='train')
    plt.plot(x_epoch, acc_data['val'], color='orange',linestyle='solid', marker='.', label='val')
    plt.legend()
    fig_acc.savefig(wdir+"out_fig/acc/{model_name}.jpg".format(model_name=args.model_name))

# Record Data in CSV file
def record_csv(x_epoch, loss_data, acc_data):
    data_dict = {'epoch': x_epoch, 'train_loss': loss_data['train'],
                 'val_loss': loss_data['val'], 'train_acc': acc_data['train'],
                 'val_acc': acc_data['val']}
    
    df = pd.DataFrame(data_dict)
    write_mode = 'w'
    if args.resume:
        write_mode = 'a'
    
    df.to_csv(wdir+"out_data/{model_name}.csv".format(model_name=args.model_name), mode = write_mode, index=False)

    




if args.test:
    test(start_epoch)
else:
    for epoch in range(start_epoch, start_epoch+150):

        train(epoch)
        test(epoch)
        x_epoch.append(epoch)
        scheduler.step()

    fig_plot(x_epoch, loss_data, acc_data)
    record_csv(x_epoch, loss_data, acc_data)


