import sys
import os
import pickle
import math

import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import *
import torchvision.transforms as transforms

# constants
LABEL_FILE = 'stage_1_train_labels.csv'
CLASS_FILE = 'stage_1_detailed_class_info.csv'
CLASS_MAPPING = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# fix for 'RuntimeError: received 0 items of ancdata' problem
if sys.platform == 'linux':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def load_dicom_image(filename):
    ds = sitk.ReadImage(filename)

    image = sitk.GetArrayFromImage(ds)
    c, w, h = image.shape

    # we know these are uint8 gray pictures already
    image = Image.fromarray(image[0])

    return image, w, h

def label_parser(filename, anno, class_info, class_mapping):
    patientId = filename.split('.')[0]

    # get class number
    class_row = class_info[class_info['patientId'] == patientId].iloc[0]
    class_name = class_row['class']
    class_no = class_mapping[class_name]

    # create label
    label = {
        'class_no': class_no,
        'bboxes': []
    }

    # read bboxes for targets
    if class_no == class_mapping['Lung Opacity']:
        anno_rows = anno[anno['patientId'] == patientId]

        for row in anno_rows:
            # [xmin, ymin, width, height]
            bbox = [
                row['x'],
                row['y'],
                row['width'],
                row['height']
            ]

            label['bboxes'].append(bbox)

    return patientId, label


class RsnaDataset(Dataset):
    def __init__(self, root, class_mapping=CLASS_MAPPING, num_classes = 3, phase='train', transforms=None):
        self.root = root
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.phase = phase
        self.transforms = transforms

        # list images
        image_path = os.path.join(self.root, self.phase)
        filenames = os.listdir(image_path)

        # read labels
        if phase != 'test':
            anno_path = os.path.join(self.root, LABEL_FILE)
            self.anno = pd.read_csv(anno_path)

            class_path = os.path.join(self.root, CLASS_FILE)
            self.class_info = pd.read_csv(class_path)

            self.labels = {}
            self.bbox_count = 0

            print('Dataset phase {}, parsing labels...'.format(self.phase, len(filenames)))

            for filename in tqdm(filenames):
                patientId, label = label_parser(
                    filename,
                    self.anno,
                    self.class_info,
                    self.class_mapping
                )
                self.labels[patientId] = label
                self.bbox_count += len(label['bboxes'])

            # 1 positive bbox
            # 1 negative bbox in the same pic as positive bbox
            # num_classes - 1 negative bboxes (in non-target pics, 1 for each class)
            self.total_len = len(self.bbox_count) * (self.num_classes + 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        pass
