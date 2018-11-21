# -*- coding: utf-8 -*-
import sys
import os
import pickle
import argparse
import pickle
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from densenet import *
from datasets.rsna import *
from utils.plot import *
from utils.log import *
from loss import compute_losses
from score import ScoreCounter

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
LOG_SIZE = 512 * 1024 * 1024 # 512M
LOGGER_NAME = 'eval'

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

def calc_transfer_matrix(a, b, num_states):
    transfer_matrix = np.zeros((num_states, num_states))
    count = a.shape[0]

    for i in range(count):
            transfer_matrix[a[i]][b[i]] += 1

    transfer_matrix /= a.shape[0]

    return transfer_matrix

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Verifier Evaluation')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--parallel', action='store_true', help='run with multiple GPUs')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--log', default='./net.log', help='log file path')
parser.add_argument('--log_level', default=logging.DEBUG, type=int, help='log level')
flags = parser.parse_args()

logger = setup_logger(LOGGER_NAME, flags.log, LOG_SIZE, flags.log_level)
score = ScoreCounter()
device = torch.device(flags.device)

# image transforms
transformation = transforms.Resize(
    size=IMAGE_SIZE,
    interpolation=Image.NEAREST
)

valSet = RsnaDataset(
    root=flags.root,
    phase='eval',
    augment=None,
    transform=transformation,
    logger_name=LOGGER_NAME
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

# load scores
score.add_scores(torch.from_numpy(checkpoint['scores']))
avg_score = float(score.get_avg_score())
logger.info('Scores loaded, avg: {}'.format(avg_score))

def eval():
    print('\nEval')

    with torch.no_grad():
        model.eval()

        val_loss = []
        val_class_accu = []
        val_score_accu = []
        val_gts = []
        val_results = []

        # perfrom forward
        for images, classes, scores, ws, hs, ids in tqdm(valLoader):
            images = images.to(device)

            classes = classes.to(device=device)
            scores = scores.to(device=device)
            gts = (classes, scores)

            # forward
            (class_outputs, score_outputs) = model(images)

            # get class accu
            results = F.softmax(class_outputs, dim=-1) # shape = [N, 4]
            results = torch.argmax(results, dim=-1) # shape = [N]

            results = results.to(dtype=torch.long)
            class_measures = torch.eq(results, classes).to(dtype=torch.float32)

            val_class_accu.append(class_measures.detach())

            # get score accu
            gt_score_results = torch.gt(scores, 0.7)
            output_score_results = torch.gt(score_outputs.squeeze(), avg_score)
            score_measures = torch.eq(output_score_results, gt_score_results).to(dtype=torch.float32)
            # print('\n{}\n{}\n{}\n'.format(gt_score_results, output_score_results, score_measures))

            val_score_accu.append(score_measures.detach())

            # loss
            class_loss, score_loss = compute_losses((class_outputs, score_outputs), gts)
            loss = class_loss + score_loss

            val_loss.append(loss.detach())

            val_gts.append(gts)
            val_results.append(results)
        
        val_class_accu = torch.cat(val_class_accu)
        val_score_accu = torch.cat(val_score_accu)
        val_loss = torch.tensor(val_loss)

        avg_class_accu = val_class_accu.mean().item()
        avg_score_accu = val_score_accu.mean().item()
        avg_loss = val_loss.mean().item()

        print('avg_loss:{}\navg_cls_accu: {}\navg_score_accu: {}'.format(
            avg_loss,
            avg_class_accu,
            avg_score_accu
        ))

        '''
        val_gts = torch.cat(val_gts)
        val_results = torch.cat(val_results)

        matrix = calc_transfer_matrix(
            val_gts.cpu().numpy(),
            val_results.cpu().numpy(),
            num_classes
        )

        f, ax = plt.subplots(figsize=(num_classes, num_classes))
        sns.heatmap(matrix, annot=True, linewidths=.5, ax=ax)
        plt.show()
        '''

# ok, main loop
if __name__ == '__main__':
    eval()
