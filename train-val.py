# -*- coding: utf-8 -*-
import sys
import os
import pickle
import argparse
import pickle
import itertools

from densenet import *
from datasets.rsna import *
from utils.plot import *

import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

# constants & configs
num_classes = 4
snapshot_interval = 5000

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 2

# variables
start_epoch = 0
best_loss = float('inf')

# helper functions
def xavier(param):
    init.xavier_uniform_(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def snapshot(epoch, batch, model, loss):
    print('Taking snapshot, loss: {}'.format(loss))

    state = {
        'net': model.state_dict(),
        'loss': loss,
        'epoch': epoch
    }
    
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')

    torch.save(state, './snapshot/e_{:0>6}_b_{:0>6}_loss_{:.5f}.pth'.format(
        epoch,
        batch,
        loss
    ))

def encode_gt(gts):
    result = []

    for gt in gts:
        if gt[0] == 0 and gt[1] == 0:
            result.append(0) # negative in 'Normal'
        elif gt[0] == 1 and gt[1] == 0:
            result.append(1) # negative in 'No Lung Opacity / Not Normal'
        elif gt[0] == 2 and gt[1] == 0:
            result.append(2) # negative in 'Lung Opacity'
        elif gt[0] == 2 and gt[1] == 1:
            result.append(3) # positive
        else:
            result.append(-1) # this is not what we want

    return result


# argparser
parser = argparse.ArgumentParser(description='Pneumonia Verifier Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--transfer', action='store_true', help='fintune pretrained model')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot loss and accuracy')
flags = parser.parse_args()

device = torch.device(flags.device)

# augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

trainSet = RsnaDataset(
    root=flags.root,
    phase='train',
    transforms=augmentation
)

trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=rsna_collate
)

valSet = RsnaDataset(
    root=flags.root,
    phase='val',
    transforms=None
)

valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=rsna_collate
)

# model
model = DenseNet121(num_classes)
model.to(device)

if flags.resume:
    # TODO : load history
    checkpoint = torch.load(flags.checkpoint)
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['net'])
elif flags.transfer:
    checkpoint = torch.load(flags.checkpoint)
    model.transfer(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss(reduce=False)

optimizer = optim.SGD(
    model.parameters(),
    lr=flags.lr,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.2,
    patience=20,
    verbose=True
)

def train(epoch):
    print('\nTraining Epoch: {}'.format(epoch))

    model.train()
    train_loss = 0
    batch_count = len(trainLoader)

    for batch_index, (images, gts, ws, hs, ids) in enumerate(trainLoader):
        images = images.to(device)

        gts = encode_gt(gts)
        gts = torch.tensor(gts, device=device, dtype=torch.long)

        optimizer.zero_grad()

        if torch.cuda.device_count() > 1:
            outputs = nn.parallel.data_parallel(model, images)
        else:
            outputs = model(images)
        
        if torch.cuda.device_count() > 1:
            loss = nn.parallel.data_parallel(criterion, (outputs, gts))
        else:
            loss = criterion(outputs, gts)

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
            epoch,
            flags.end_epoch - 1,
            batch_index,
            batch_count - 1,
            loss.item(),
            train_loss / (batch_index + 1)
        ))

        if (batch_index + 1) % snapshot_interval == 0:
            snapshot(epoch, batch_index, model, train_loss / (batch_index + 1))

def val(epoch):
    print('\nVal')

    with torch.no_grad():
        model.eval()
        val_loss = 0
        batch_count = len(valLoader)

        # perfrom forward
        for batch_index, (images, gts, ws, hs, ids) in enumerate(valLoader):
            images = images.to(device)

            gts = encode_gt(gts)
            gts = torch.tensor(gts, device=device, dtype=torch.long)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(model, images)
            else:
                outputs = model(images)

            if torch.cuda.device_count() > 1:
                loss = nn.parallel.data_parallel(criterion, (outputs, gts))
            else:
                loss = criterion(outputs, gts)

            loss = loss.mean()

            val_loss += loss.item()

            print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
                epoch,
                flags.end_epoch - 1,
                batch_index,
                batch_count - 1,
                loss.item(),
                val_loss / (batch_index + 1)
            ))

        global best_loss
        val_loss /= batch_count

        # update lr
        scheduler.step(val_loss)

        # save checkpoint
        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(val_loss))

            state = {
                'net': model.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
            }
            
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/epoch_{:0>5}_loss_{:.5f}.pth'.format(
                epoch,
                val_loss
            ))

            best_loss = val_loss

# ok, main loop
if __name__ == '__main__':
    for epoch in range(start_epoch, flags.end_epoch):
        scheduler.step(best_loss)
        train(epoch)
        val(epoch)
