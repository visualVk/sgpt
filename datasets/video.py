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


class VIDEO(torch.utils.data.Dataset):

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
        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['img_id'] = int(
            img_anno['file_name'].rstrip('.jpg').split('_')[0])

        if self._transforms is not None:
            img, _ = self._transforms(img, None)

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
        'val': (root / 'frames', root / 'img_anno.json')
    }
    # CORRECT_MAT_PATH = root / 'annotations' / 'corre_vcoco.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = VIDEO(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set),
                    num_queries=args.num_queries)
    # if image_set == 'val':
    #     dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
