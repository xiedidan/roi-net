import sys
import os
import argparse
import pickle

from datasets.rsna import *
from utils.plot import *

import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

# config
IMAGE_SIZE = 256

# argparser
parser = argparse.ArgumentParser(description='RSNA dataset toolkit')
parser.add_argument('--root', default='./rsna/', help='dataset root path')
flags = parser.parse_args()

if __name__ == '__main__':
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

    # bbox augmentation
    bbox_augmentation = iaa.Sequential([
        iaa.OneOf([
            # geometry transform - keep lowest IOU around 0.8
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            ),
            iaa.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            )
        ])
    ])

    transformation = transforms.Resize(
        size=IMAGE_SIZE,
        interpolation=Image.NEAREST
    )

    # data
    trainSet = RsnaDataset(
        root=flags.root,
        phase='train',
        augment=augmentation,
        bbox_augment=bbox_augmentation,
        transform=transformation
    )

    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=rsna_collate
    )

    for images, gts, ws, hs, ids in tqdm(trainLoader):
        for gt, patientId in zip(gts, ids):
            print('patientId: {}, global: {}, roi: {}'.format(
                patientId,
                list(trainSet.class_mapping.keys())[gt[0]],
                'hit' if gt[1] == 1 else 'miss'
            ))

        plot_batch(images)
