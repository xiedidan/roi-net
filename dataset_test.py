import sys
import os
import argparse
import pickle

from datasets.rsna import *
from utils.plot import *

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

# argparser
parser = argparse.ArgumentParser(description='RSNA dataset toolkit')
parser.add_argument('--root', default='./rsna/', help='dataset root path')
flags = parser.parse_args()

if __name__ == '__main__':
    # data
    trainSet = RsnaDataset(
        root=flags.root,
        phase='train',
        transforms=None
    )

    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        collate_fn=rsna_collate
    )

    for images, gts, ws, hs, ids in tqdm(trainLoader):
        for gt, patientId in zip(gts.numpy(), ids):
            print('patientId: {}, global: {}, roi: {}'.format(
                patientId,
                list(trainSet.class_mapping.keys())[gt[0]],
                'hit' if gt[1] == 1 else 'miss'
            ))

        plot_batch(images)
