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

from PIL import Image
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
IMAGE_SIZE = 512

# on linux, each worker could take 300% of cpu
# so 2 workers are enough for a 4 core machine
number_workers = 2
if sys.platform == 'win32':
    # spawned workers on windows take too much gmem
    # so keep lower number of workers
    number_workers = 2

# variables

# helper functions
def xavier(param):
    init.xavier_uniform_(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

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
parser = argparse.ArgumentParser(description='Pneumonia Verifier Evaluation')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--parallel', action='store_true', help='run with multiple GPUs')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

device = torch.device(flags.device)

# image transforms
transformation = transforms.Resize(
    size=IMAGE_SIZE,
    interpolation=Image.NEAREST
)

valSet = RsnaDataset(
    root=flags.root,
    phase='val',
    augment=None,
    transform=transformation
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

# load checkpoint
checkpoint = torch.load(flags.checkpoint)
model.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss(reduce=False)

def eval():
    print('\nEval')

    with torch.no_grad():
        model.eval()

        val_loss = 0.
        val_accu = 0.
        batch_count = len(valLoader)

        # perfrom forward
        for batch_index, (images, gts, ws, hs, ids) in enumerate(valLoader):
            images = images.to(device)

            gts = encode_gt(gts)
            gts = torch.tensor(gts, device=device, dtype=torch.long)

            # forward
            if torch.cuda.device_count() > 1 and flags.parallel:
                outputs = nn.parallel.data_parallel(model, images)
            else:
                outputs = model(images)

            # get accu
            results = F.softmax(outputs, dim=-1) # shape = [N, 4]
            results = torch.argmax(results, dim=-1) # shape = [N]

            results = results.to(dtype=torch.long)
            scores = torch.eq(results, gts).to(dtype=torch.float32)

            score = scores.mean()
            val_accu += score.item()

            # loss
            if torch.cuda.device_count() > 1 and flags.parallel:
                loss = nn.parallel.data_parallel(criterion, (outputs, gts))
            else:
                loss = criterion(outputs, gts)

            loss = loss.mean()
            val_loss += loss.item()

            print('b:{}/{}, b_l:{:.2f}, avg_l:{:.2f}, b_a:{:.2f}, avg_a:{:.2f}'.format(
                batch_index,
                batch_count - 1,
                loss.item(),
                val_loss / (batch_index + 1),
                score.item(),
                val_accu / (batch_index + 1)
            ))

        val_loss /= batch_count
        val_accu /= batch_count

        print('avg_loss:{:.2f}, avg_accuracy: {:.2f}'.format(
            val_loss,
            val_accu
        ))

# ok, main loop
if __name__ == '__main__':
    eval()
