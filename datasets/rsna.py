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
IOU_THRESHOLD = 0.2
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

    # read bboxes for targets
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
def generate_percent_bbox(x_min, x_max, y_min, y_max):
    w = random.uniform(x_min, x_max)
    h = random.uniform(y_min, y_max)

    x_range = 1. - w
    y_range = 1. - h

    xmin = random.uniform(0., x_range)
    ymin = random.uniform(0., y_range)

    return [xmin, ymin, w, h]

def to_pointform(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3]
    ]

def to_bbox(pointform):
    return [
        pointform[0],
        pointform[1],
        pointform[2] - pointform[0],
        pointform[3] - pointform[1]
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
    def __init__(self, root, class_mapping=CLASS_MAPPING, num_classes = 3, phase='train', transforms=None):
        self.root = root
        self.class_mapping = class_mapping
        self.num_classes = num_classes
        self.phase = phase
        self.transforms = transforms

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

            label = copy.deepcopy(self.labels[class_index])

            if class_no < self.num_classes:
                if class_no == self.class_mapping['Lung Opacity']:
                    # directly use the label
                    # load image
                    image, w, h = load_dicom_image(os.path.join(
                        self.root,
                        self.phase,
                        '{}.dcm'.format(label['patientId'])
                    ))

                    label['bbox'] = to_percent(label['bbox'], w, h)

                    roi_class = 1
                else:
                    # use bbox in the label to crop from a randomly selected image
                    class_patients = self.non_targets[class_no]
                    patientId = random.choice(class_patients)

                    label['patientId'] = patientId

                    # load image
                    image, w, h = load_dicom_image(os.path.join(
                        self.root,
                        self.phase,
                        '{}.dcm'.format(label['patientId'])
                    ))

                    label['bbox'] = to_percent(label['bbox'], w, h)

                    roi_class = 0
            else:
                # pick a negative bbox from the target
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

                while True: # TODO : limited trails?
                    new_bbox = generate_percent_bbox(0.007, 0.4, 0.007, 0.8)
                    # print(patient_bboxes, new_bbox)
                    sys.stdout.flush()
                    iou = jaccard_numpy(
                        np.array([to_pointform(bbox) for bbox in patient_bboxes]),
                        np.array(to_pointform(new_bbox))
                    )

                    if (iou < IOU_THRESHOLD).all():
                        break

                label['bbox'] = new_bbox

                roi_class = 0

            # bounding box transformation
            absolute_bbox = to_absolute(label['bbox'], w, h)
            pt_form = to_pointform(absolute_bbox)

            # imgaug transform
            if self.transforms is not None:
                transforms_det = self.transforms.to_deterministic()

                np_image = np.asarray(image)
                np_image = np_image[:, :, np.newaxis]

                bbs = ia.BoundingBoxesOnImage(
                    [ ia.BoundingBox(x1=pt_form[0], y1=pt_form[1], x2=pt_form[2], y2=pt_form[3]) ],
                    shape=np_image.shape
                )

                # transforms
                np_image_aug = transforms_det.augment_images([np_image])[0]
                bbs_aug = transforms_det.augment_bounding_boxes([bbs])[0]
                bbs_aug = bbs_aug.remove_out_of_image()

                # if there's no bbox after augmentation
                # just fallback to original sample
                if len(bbs_aug.bounding_boxes) > 0:
                    # convert to PIL image
                    image = Image.fromarray(np.array(np_image_aug).squeeze())

                    # populate bbox represents
                    pt_bbox = bbs_aug.bounding_boxes[0]
                    pt_form = [pt_bbox.x1_int, pt_bbox.y1_int, pt_bbox.x2_int, pt_bbox.y2_int]
                    absolute_bbox = to_bbox(pt_form)
                    label['bbox'] = to_percent(absolute_bbox, w, h)

            # create mask - 255(uint8) and 1.0(float32) are both ok
            mask = np.zeros((h, w, 1), dtype=np.uint8)
            mask[pt_form[1]:pt_form[3], pt_form[0]:pt_form[2], :] = 255

            # crop & resize
            crop = transforms.functional.resized_crop(
                image,
                absolute_bbox[1],
                absolute_bbox[0],
                absolute_bbox[3],
                absolute_bbox[2],
                (h, w)
            )

            # create tensor and cat image layers - [c, h, w]
            image = transforms.functional.to_tensor(image)
            mask = transforms.functional.to_tensor(mask)
            crop = transforms.functional.to_tensor(crop)

            layers = torch.cat((image, mask, crop), dim=0)

            # gt
            global_class = class_no if class_no < self.num_classes else self.num_classes - 1
            gt = [global_class, roi_class]

            return layers, gt, w, h, label['patientId']
