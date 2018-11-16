import sys
import os
import pickle
import math
import random
import copy

import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

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
IOU_MIN_THRESHOLD = 0.4
IOU_MAX_THRESHOLD = 0.75
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 3
IA_SEED = 1

# fix for 'RuntimeError: received 0 items of ancdata' problem
if sys.platform == 'linux':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# helpers
def load_dicom_image(filename):
    ds = sitk.ReadImage(filename)

    image = sitk.GetArrayFromImage(ds) # z, y, x
    c, h, w = image.shape

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
    label = []

    # read absolute corner form bbox for targets
    if class_no == class_mapping['Lung Opacity']:
        anno_rows = anno[anno['patientId'] == patientId]

        for index, row in anno_rows.iterrows():
            # [xmin, ymin, width, height]
            bbox = [
                row['x'],
                row['y'],
                row['width'],
                row['height']
            ]

            label.append({
                'patientId': patientId,
                'bbox': bbox
            })

    return label, class_no

# x, y range in percentage
def generate_percent_corner(x_min, x_max, y_min, y_max):
    w = random.uniform(x_min, x_max)
    h = random.uniform(y_min, y_max)

    x_range = 1. - w
    y_range = 1. - h

    xmin = random.uniform(0., x_range)
    ymin = random.uniform(0., y_range)

    return [xmin, ymin, w, h]

def transform_percent_corner(p_cn):
    x = random.uniform(p_cn[0] - p_cn[2] * 0.176, p_cn[0] + p_cn[2] * 0.176)
    y = random.uniform(p_cn[1] - p_cn[3] * 0.176, p_cn[1] + p_cn[3] * 0.176)

    w = random.uniform(p_cn[2] * 0.7, p_cn[2] * 1.4)
    h = random.uniform(p_cn[3] * 0.7, p_cn[3] * 1.4)

    return [x, y, w, h]

def to_point(corner):
    return [
        corner[0],
        corner[1],
        corner[0] + corner[2],
        corner[1] + corner[3]
    ]

def to_corner(point):
    return [
        point[0],
        point[1],
        point[2] - point[0],
        point[3] - point[1]
    ]

def to_percent(bbox, width, height):
    return [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height
    ]

def to_absolute(bbox, width, height):
    return [
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height)
    ]

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def rsna_collate(batch):
    images = []
    gts = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample

        images.append(image)
        gts.append(gt)
        ws.append(w)
        hs.append(h)
        ids.append(patientId)

    images = torch.stack(images, dim=0) # now, [n, c, h, w]

    return images, gts, ws, hs, ids

class RsnaDataset(Dataset):
    def __init__(self, root, class_mapping=CLASS_MAPPING, num_classes = 3, phase='train', augment=None, bbox_augment=None, transform=None):
        self.root = root
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.phase = phase
        self.augments = augment
        self.bbox_augments = bbox_augment
        self.transforms = transform

        ia.seed(IA_SEED)

        # list images
        image_path = os.path.join(self.root, self.phase)
        filenames = os.listdir(image_path)

        # read labels
        if self.phase != 'test':
            anno_path = os.path.join(self.root, LABEL_FILE)
            self.anno = pd.read_csv(anno_path)

            class_path = os.path.join(self.root, CLASS_FILE)
            self.class_info = pd.read_csv(class_path)

            self.labels = []
            self.non_targets = [[] for _ in range(self.num_classes)]

            print('Dataset phase {}'.format(self.phase))

            # get all target boxes
            print('Parsing targets:')
            target_rows = self.class_info[self.class_info['class'] == 'Lung Opacity']
            
            with tqdm(total=len(target_rows)) as pbar:
                for i, target_row in target_rows.iterrows():
                    pbar.update(1)

                    filename = '{0}.dcm'.format(target_row['patientId'])
                    if filename in filenames:
                        label, class_no = label_parser(
                            filename,
                            self.anno,
                            self.class_info,
                            self.class_mapping
                        )
                        self.labels.extend(label)

            # non-targets
            print('Parsing non-targets:')
            non_target_rows = self.class_info[self.class_info['class'] != 'Lung Opacity']

            with tqdm(total=len(non_target_rows)) as pbar:
                for i, non_target_row in tqdm(non_target_rows.iterrows()):
                    pbar.update(1)
                    
                    if '{0}.dcm'.format(non_target_row['patientId']) in filenames:
                        class_no = self.class_mapping[non_target_row['class']]
                        self.non_targets[class_no].append(non_target_row['patientId'])

            # 1 positive bbox
            # 1 negative bbox in the same pic as positive bbox
            # num_classes - 1 negative bboxes (in non-target pics, 1 for each class)
            self.total_len = len(self.labels) * (self.num_classes + 1)
            
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.phase != 'test':
            class_no = index // len(self.labels)
            class_index = index % len(self.labels)

            label = copy.deepcopy(self.labels[class_index]) # for now, label is a_cn

            if class_no < self.num_classes:
                if class_no == self.class_mapping['Lung Opacity']:
                    # load image
                    image, w, h = load_dicom_image(os.path.join(
                        self.root,
                        self.phase,
                        '{}.dcm'.format(label['patientId'])
                    ))

                    # get target bboxes
                    patient_annos = self.anno[self.anno['patientId'] == label['patientId']]
                    patient_label, class_no = label_parser(
                        '{0}.dcm'.format(label['patientId']),
                        patient_annos,
                        self.class_info,
                        self.class_mapping
                    )
                    patient_bboxes = [to_percent(label['bbox'], w, h) for label in patient_label]

                    # randomly generate a bbox
                    while True: # TODO : limited trails?
                        new_bbox = transform_percent_corner(to_percent(label['bbox'], w, h))
                        iou = jaccard_numpy(
                            np.array([to_point(bbox) for bbox in patient_bboxes]),
                            np.array(to_point(new_bbox))
                        )
                        aspect_ratio = new_bbox[2] / new_bbox[3]

                        if (iou > IOU_MAX_THRESHOLD).any() and (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
                            break

                    label['bbox'] = new_bbox

                    roi_class = 1
                else:
                    # randomly select a negative image                    
                    class_patients = self.non_targets[class_no]
                    patientId = random.choice(class_patients)

                    label['patientId'] = patientId

                    # load image
                    image, w, h = load_dicom_image(os.path.join(
                        self.root,
                        self.phase,
                        '{}.dcm'.format(label['patientId'])
                    ))

                    # randomly generate a bbox
                    while True: # TODO : limited trails?
                        new_bbox = generate_percent_corner(0.007, 0.4, 0.007, 0.8)
                        aspect_ratio = new_bbox[2] / new_bbox[3]

                        if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                            break

                    label['bbox'] = new_bbox

                    roi_class = 0
            else:
                # load image
                image, w, h = load_dicom_image(os.path.join(
                    self.root,
                    self.phase,
                    '{}.dcm'.format(label['patientId'])
                ))

                # get target bboxes
                patient_annos = self.anno[self.anno['patientId'] == label['patientId']]
                patient_label, class_no = label_parser(
                    '{0}.dcm'.format(label['patientId']),
                    patient_annos,
                    self.class_info,
                    self.class_mapping
                )
                patient_bboxes = [to_percent(label['bbox'], w, h) for label in patient_label]

                # randomly generate a bbox
                while True: # TODO : limited trails?
                    new_bbox = generate_percent_corner(0.007, 0.4, 0.007, 0.8)
                    iou = jaccard_numpy(
                        np.array([to_point(bbox) for bbox in patient_bboxes]),
                        np.array(to_point(new_bbox))
                    )
                    aspect_ratio = new_bbox[2] / new_bbox[3]

                    if (iou < IOU_MIN_THRESHOLD).all() and (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
                        break

                label['bbox'] = new_bbox

                roi_class = 0

            # image transforms
            if self.transforms is not None:
                new_image = self.transforms(image)
                new_w, new_h = new_image.size # Image.size returns (w, h)
            else:
                new_image = image
                new_w = w
                new_h = h

            # bounding box transformation - label['bbox'] is p_cn
            new_a_cn = to_absolute(label['bbox'], new_w, new_h)
            new_a_pt = to_point(new_a_cn)

            # imgaug augments
            if self.augments is not None:
                augments_det = self.augments.to_deterministic()

                np_image = np.asarray(new_image)
                np_image = np_image[:, :, np.newaxis]

                bbs = ia.BoundingBoxesOnImage(
                    [ ia.BoundingBox(x1=new_a_pt[0], y1=new_a_pt[1], x2=new_a_pt[2], y2=new_a_pt[3]) ],
                    shape=np_image.shape
                )

                # transforms
                np_image_aug = augments_det.augment_images([np_image])[0]

                bbs_aug = augments_det.augment_bounding_boxes([bbs])[0]
                bbs_aug = bbs_aug.remove_out_of_image()

                # if there's no bbox after augmentation
                # just fallback to original sample
                if len(bbs_aug.bounding_boxes) > 0:
                    # convert to PIL image
                    new_image = Image.fromarray(np.array(np_image_aug).squeeze())

                    # populate bbox represents
                    new_a_pt_aug = bbs_aug.bounding_boxes[0]
                    new_a_pt = [new_a_pt_aug.x1_int, new_a_pt_aug.y1_int, new_a_pt_aug.x2_int, new_a_pt_aug.y2_int]
                    new_a_cn = to_corner(new_a_pt)

                    label['bbox'] = to_percent(new_a_cn, new_w, new_h) # p_cn

            # create mask - 255(uint8) and 1.0(float32) are both ok
            mask = np.zeros((new_h, new_w, 1), dtype=np.uint8)
            mask[new_a_pt[1]:new_a_pt[3], new_a_pt[0]:new_a_pt[2], :] = 255

            # crop & resize
            crop = transforms.functional.resized_crop(
                new_image,
                new_a_cn[1],
                new_a_cn[0],
                new_a_cn[3],
                new_a_cn[2],
                (new_h, new_w)
            )

            # create tensor and cat image layers - [c, h, w]
            new_image = transforms.functional.to_tensor(new_image)
            mask = transforms.functional.to_tensor(mask)
            crop = transforms.functional.to_tensor(crop)

            layers = torch.cat((new_image, mask, crop), dim=0)

            # gt
            global_class = class_no if class_no < self.num_classes else self.num_classes - 1
            gt = [global_class, roi_class]

            return layers, gt, w, h, label['patientId']
