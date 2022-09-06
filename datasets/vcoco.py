# ------------------------------------------------------------------------
# QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image
import json
from loguru import logger
import numpy as np

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T


class VCOCO(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        logger.info(f"==> {anno_file}")
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self.kpt_ids = torch.arange(17)

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = range(29)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]
        img = Image.open(self.img_folder /
                         img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        keypoints = [obj['keypoint'] for obj in img_anno['annotations']]
        center = [obj['center'] for obj in img_anno['annotations']]

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id']))
                       for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(
                obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['joint_boxes'] = torch.tensor(
                keypoints, dtype=torch.float32).view(-1, 17, 3)
            target['center'] = torch.tensor(center)
            target['iscrowd'] = torch.tensor(
                [0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * \
                (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            # sub_keypoints, kpt_labels = [], []
            sub_keypoints = target['joint_boxes'][:, :, :2]
            kpt_labels = target['joint_boxes'][:, :, -1] > 0.0
            sub_ids = []
            sub_obj_pairs = []
            # center_points = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or \
                   (hoi['object_id'] != -1 and hoi['object_id'] not in kept_box_indices):
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(
                        sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_ids.append(torch.tensor(kept_box_indices.index(sub_obj_pair[0])))
                    sub_obj_pairs.append(sub_obj_pair)
                    if hoi['object_id'] == -1:
                        obj_labels.append(torch.tensor(
                            len(self._valid_obj_ids)))
                    else:
                        obj_labels.append(
                            target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(
                        hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(
                        hoi['subject_id'])]
                    # sub_keypoint = target['joint_boxes'][kept_box_indices.index(
                    #     hoi['subject_id'])]
                    # kpt_label = torch.cat([sub_keypoint[:, -1] > 0.0])
                    # sub_keypoint = sub_keypoint[:, :2]
                    # sub_keypoints.append(sub_keypoint)
                    if hoi['object_id'] == -1:
                        obj_box = torch.zeros((4,), dtype=torch.float32)
                    else:
                        obj_box = target['boxes'][kept_box_indices.index(
                            hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                    # kpt_labels.append(kpt_label)
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros(
                    (0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['joint_boxes'] = torch.zeros(
                    (0, 17, 2), dtype=torch.float32)
                target['kpt_labels'] = torch.zeros((0, 17), dtype=torch.bool)
                target['sub_ids'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(
                    verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['sub_ids'] = torch.stack(sub_ids)
                target['joint_boxes'] = sub_keypoints
                target['kpt_labels'] = kpt_labels
                # target['joint_boxes'] = torch.stack(sub_keypoints)
                # target['kpt_labels'] = torch.stack(kpt_labels)
        else:
            target['boxes'] = boxes
            target['joint_boxes'] = torch.tensor(
                keypoints, dtype=torch.float32).view(-1, 17, 3)
            target['labels'] = classes
            target['id'] = idx
            target['num_keypoints'] = torch.tensor([anno['num_keypoints']
                                       for anno in img_anno['annotations']], dtype=torch.int64)
            target['ids'] = torch.tensor([anno['id'] for anno in img_anno['annotations']], dtype=torch.int64)
            target['img_id'] = int(
                img_anno['file_name'].rstrip('.jpg').split('_')[2])

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois, sub_boxes, sub_ids = [], [], []
            joint_boxes = target['joint_boxes'][:, :, :2]
            kpt_labels = target['joint_boxes'][:, :, -1] > 0.0
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'],
                            self._valid_verb_ids.index(hoi['category_id'])))
                # joint_box = target['joint_boxes'][hoi['subject_id']].tolist()
                # kpt_label = torch.cat([torch.tensor(self.kpt_ids[s[:, -1] > 0.0]) for s in joint_boxes])
                # joint_boxes.append(joint_box[:, :, :2])
                # kpt_labels.append(kpt_label)
                sub_ids.append(hoi['subject_id'])
                sub_boxes.append(target['boxes'][hoi['subject_id']].tolist())

            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
            target['joint_boxes'] = torch.as_tensor(
                joint_boxes, dtype=torch.float32)
            target['sub_boxes'] = torch.as_tensor(
                sub_boxes, dtype=torch.float32)
            target['sub_ids'] = torch.as_tensor(sub_ids, dtype=torch.int64)
            target['kpt_labels'] = torch.as_tensor(
                kpt_labels, dtype=torch.bool)

        return img, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]),
            ),
            # T.RandomResize(scales, max_size=1333),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2014', root / 'annotations' / 'trainval_vcoco_center.json'),
        'val': (root / 'images' / 'val2014', root / 'annotations' / 'test_vcoco_center.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_vcoco.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = VCOCO(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set),
                    num_queries=args.num_queries)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
