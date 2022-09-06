import json
import numpy as np
import json
import pandas as pd
import os
from PIL import Image
from os import path as osp
import torch

from typing import Dict, Any

from tqdm import tqdm

def parser_vcoco_log(file_path, selected_keys=['test_mAP_all', 'test_mAP_thesis', 'train_loss']):
    return parser_log(file_path, selected_keys)


def parser_hico_log(file_path, selected_keys=['train_loss', 'test_mAP', 'test_mAP rare', 'test_mAP non-rare', 'test_mean max recall']):
    return parser_log(file_path, selected_keys)


def parser_log(file_path, selected_keys=['train_loss']):
    res = {k: [] for k in selected_keys}
    with open(file_path) as f:
        for line in f.readlines():
            logs = json.loads(line)
            for k in selected_keys:
                res[k].append(logs[k])
    return res


def add_suffix_for_results(res: Dict[str, Any], suffix=None):
    if suffix is None:
        return
    return {k + f'_{suffix}': v for k, v in res.items()}


def init_plt(plt, size=(16, 8), interpolate='nearest', cmap='gray'):
    plt.rcParams['figure.figsize'] = size  # 设置figure_size尺寸
    plt.rcParams['image.interpolation'] = interpolate  # 设置 interpolation style
    plt.rcParams['image.cmap'] = cmap  # 设置 颜色 style
    return plt


def end_plt(plt, title, grid=True, legend=True, tot_len=None, step=5):
    plt.title(title)
    plt.grid(grid)
    if legend:
        plt.legend()
    if tot_len is not None:
        plt.xstick(np.arange(0, tot_len, step=step))
    return plt


def print_max_value(res, selected_labels):
    for label in selected_labels:
        max_comb = torch.tensor(res[label]).max(dim=-1)
        print(
            f"metric name: {label}, value: {max_comb[0]}, indice: {max_comb[1]}")


def print_index_selected_value(res, selected_labels, idx):
    for label in selected_labels:
        max_comb = torch.tensor(res[label][idx])
        print(f"metric name: {label}, value: {max_comb}")


def save_svg_from_plt(plt, filename):
    plt.savefig(filename)


def preds_gts_with_boxes(preds, gts):
    # preds = preds_baseline
    # gts = gts_baseline
    preds_hoi_filtered = []
    for pred, gt in zip(preds, gts):
        pred_boxes = pred['predictions']
        pred_hoi = pred['hoi_prediction']
        gt_boxes = gt['annotations']
        gt_hoi = gt['hoi_annotation']

        cat_ids = []
        for hoi in gt_hoi:
            vid = hoi['category_id']
            oid = gt_boxes[hoi['object_id']]['category_id']
            cat_ids.append((oid, vid))

        hois = []
        for hoi in pred_hoi:
            vid = hoi['category_id']
            oid = pred_boxes[hoi['object_id']]['category_id']
            # hid = pred_boxes[hoi['subject_id']]['category_id']
            if (oid, vid) in cat_ids:
                hoi_idx = hois_map.get((oid, vid), 0)
                box_h = pred_boxes[hoi['subject_id']]['bbox']
                box_o = pred_boxes[hoi['object_id']]['bbox']
                hoi['box_h'] = box_h
                hoi['box_o'] = box_o
                hoi['hoi_idx'] = hoi_idx
                hois.append(hoi)
        preds_hoi_filtered.append(hois)

    gts_with_boxes = []
    for gt in gts:
        hois = []
        gt_boxes = gt['annotations']
        gt_hoi = gt['hoi_annotation']
        for hoi in gt_hoi:
            vid = hoi['category_id']
            oid = gt_boxes[hoi['object_id']]['category_id']
            hoi_idx = hois_map.get((oid, vid), 0)
            box_h = gt_boxes[hoi['subject_id']]['bbox']
            box_o = gt_boxes[hoi['object_id']]['bbox']
            hoi['box_h'] = box_h
            hoi['box_o'] = box_o
            hoi['hoi_idx'] = hoi_idx
            hois.append(hoi)
        gts_with_boxes.append(hois)

    return preds_hoi_filtered, gts_with_boxes


def get_hoi_map(hoi_map_path='data/hico_20160224_det/add_file/hois_id.csv'):
    hois_id = pd.read_csv(hoi_map_path)
    hois_id_list = hois_id.to_numpy().tolist()
    hois_map = {(v[1], v[2]): v[0] for v in hois_id_list}
    return hois_map


def get_verb_map():
    vb_map = {'adjust': 0,
              'assemble': 1,
              'block': 2,
              'blow': 3,
              'board': 4,
              'break': 5,
              'brush_with': 6,
              'buy': 7,
              'carry': 8,
              'catch': 9,
              'chase': 10,
              'check': 11,
              'clean': 12,
              'control': 13,
              'cook': 14,
              'cut': 15,
              'cut_with': 16,
              'direct': 17,
              'drag': 18,
              'dribble': 19,
              'drink_with': 20,
              'drive': 21,
              'dry': 22,
              'eat': 23,
              'eat_at': 24,
              'exit': 25,
              'feed': 26,
              'fill': 27,
              'flip': 28,
              'flush': 29,
              'fly': 30,
              'greet': 31,
              'grind': 32,
              'groom': 33,
              'herd': 34,
              'hit': 35,
              'hold': 36,
              'hop_on': 37,
              'hose': 38,
              'hug': 39,
              'hunt': 40,
              'inspect': 41,
              'install': 42,
              'jump': 43,
              'kick': 44,
              'kiss': 45,
              'lasso': 46,
              'launch': 47,
              'lick': 48,
              'lie_on': 49,
              'lift': 50,
              'light': 51,
              'load': 52,
              'lose': 53,
              'make': 54,
              'milk': 55,
              'move': 56,
              'no_interaction': 57,
              'open': 58,
              'operate': 59,
              'pack': 60,
              'paint': 61,
              'park': 62,
              'pay': 63,
              'peel': 64,
              'pet': 65,
              'pick': 66,
              'pick_up': 67,
              'point': 68,
              'pour': 69,
              'pull': 70,
              'push': 71,
              'race': 72,
              'read': 73,
              'release': 74,
              'repair': 75,
              'ride': 76,
              'row': 77,
              'run': 78,
              'sail': 79,
              'scratch': 80,
              'serve': 81,
              'set': 82,
              'shear': 83,
              'sign': 84,
              'sip': 85,
              'sit_at': 86,
              'sit_on': 87,
              'slide': 88,
              'smell': 89,
              'spin': 90,
              'squeeze': 91,
              'stab': 92,
              'stand_on': 93,
              'stand_under': 94,
              'stick': 95,
              'stir': 96,
              'stop_at': 97,
              'straddle': 98,
              'swing': 99,
              'tag': 100,
              'talk_on': 101,
              'teach': 102,
              'text_on': 103,
              'throw': 104,
              'tie': 105,
              'toast': 106,
              'train': 107,
              'turn': 108,
              'type_on': 109,
              'walk': 110,
              'wash': 111,
              'watch': 112,
              'wave': 113,
              'wear': 114,
              'wield': 115,
              'zip': 116}
    return {v: k for k,v in vb_map.items()}


def vis_pic_with_pred_label(preds, gts, hois_map, img_root='data/hico_20160224_det/images/test2015/', save_root='baseline'):
    from matplotlib import pyplot as plt
    
    save_root = osp.join('data/hico_20160224_det/add_file/pic_pred/', save_root)
    if not osp.exists(save_root):
        os.mkdir(save_root)
    
    coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    valid_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                 82, 84, 85, 86, 87, 88, 89, 90)
    vb_map = get_verb_map()

    for pred, gt in tqdm(zip(preds, gts), ncols=120, desc=''):
        filename = pred['filename']
        pred_boxes = pred['predictions']
        pred_hoi = pred['hoi_prediction']
        gt_boxes = gt['annotations']
        gt_hoi = gt['hoi_annotation']

        img = Image.open(osp.join(img_root, f'HICO_test2015_{filename:08d}.jpg'))
        save_img_i_root = osp.join(save_root, f'test_{filename}')
        if not osp.exists(save_img_i_root):
            os.mkdir(save_img_i_root)

        cat_ids = []
        for hoi in gt_hoi:
            vid = hoi['category_id']
            oid = gt_boxes[hoi['object_id']]['category_id']
            cat_ids.append((oid, vid))

        for j, hoi in enumerate(pred_hoi):
            vid = hoi['category_id']
            oid = pred_boxes[hoi['object_id']]['category_id']
            # hid = pred_boxes[hoi['subject_id']]['category_id']
            if (oid, vid) in cat_ids:
                # hoi_idx = hois_map.get((oid, vid), 0)
                box_h = pred_boxes[hoi['subject_id']]['bbox']
                box_o = pred_boxes[hoi['object_id']]['bbox']
                score = hoi['score']
                label_o = coco_id_name_map[valid_ids[oid]]
                label_v = vb_map[vid]
                fig, ax = plt.subplots()
                rect_h = plt.Rectangle((box_h[0], box_h[1]), box_h[2] - box_h[0], box_h[3] - box_h[1], fill=False, color='r')
                rect_o = plt.Rectangle((box_o[0], box_o[1]), box_o[2] - box_o[0], box_o[3] - box_o[1], fill=False, color='g')
                ax.imshow(img)
                ax.add_patch(rect_h)
                ax.add_patch(rect_o)
                
                ax.annotate(label_o, (int(box_o[0]) + 25, int(box_o[1]) + 25), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True, fc="r"))
                ax.annotate(label_v, (int((box_o[0] + box_h[0]) / 2), int((box_o[1] + box_h[1]) / 2)), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True, fc="b"))
                ax.annotate(f'{score * 100:.2f}', (int((box_o[0] + box_h[0]) / 2) + 25, int((box_o[1] + box_h[1]) / 2) + 25), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True))
                
                fig.savefig(osp.join(save_img_i_root, f'test_{filename}_{j}.png'))
                plt.close(fig)

def vis_pic_with_gt_label(preds, gts, hois_map, img_root='data/hico_20160224_det/images/test2015/', save_root='baseline'):
    from matplotlib import pyplot as plt
    
    save_root = osp.join('data/hico_20160224_det/add_file/pic_pred/', save_root)
    if not osp.exists(save_root):
        os.mkdir(save_root)
    
    coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    valid_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                 82, 84, 85, 86, 87, 88, 89, 90)
    vb_map = get_verb_map()

    for pred, gt in tqdm(zip(preds, gts), ncols=120, desc=''):
        filename = pred['filename']
        gt_boxes = gt['annotations']
        gt_hoi = gt['hoi_annotation']

        img = Image.open(osp.join(img_root, f'HICO_test2015_{filename:08d}.jpg'))
        save_img_i_root = osp.join(save_root, f'test_{filename}')
        if not osp.exists(save_img_i_root):
            os.mkdir(save_img_i_root)

        for j, hoi in enumerate(gt_hoi):
            vid = hoi['category_id']
            oid = gt_boxes[hoi['object_id']]['category_id']
            
            box_h = gt_boxes[hoi['subject_id']]['bbox']
            box_o = gt_boxes[hoi['object_id']]['bbox']
            label_o = coco_id_name_map[valid_ids[oid]]
            label_v = vb_map[vid]
            fig, ax = plt.subplots()
            rect_h = plt.Rectangle((box_h[0], box_h[1]), box_h[2] - box_h[0], box_h[3] - box_h[1], fill=False, color='r')
            rect_o = plt.Rectangle((box_o[0], box_o[1]), box_o[2] - box_o[0], box_o[3] - box_o[1], fill=False, color='g')
            ax.imshow(img)
            ax.add_patch(rect_h)
            ax.add_patch(rect_o)
                
            ax.annotate(label_o, (int(box_o[0]) + 25, int(box_o[1]) + 25), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True, fc="r"))
            ax.annotate(label_v, (int((box_o[0] + box_h[0]) / 2), int((box_o[1] + box_h[1]) / 2)), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True, fc="b"))
                
            fig.savefig(osp.join(save_img_i_root, f'test_{filename}_{j}.png'))
            plt.close(fig)
            

def vis_pic_by_image_id_with_pred_label(preds, gts, hois_map, image_id, img_root='data/hico_20160224_det/images/test2015/', save_root='baseline', h_c='r', o_c='g'):
    from matplotlib import pyplot as plt
    
    save_root = osp.join('data/hico_20160224_det/add_file/pic_pred/', save_root)
    if not osp.exists(save_root):
        os.mkdir(save_root)
    
    coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    valid_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                 82, 84, 85, 86, 87, 88, 89, 90)
    vb_map = get_verb_map()
    
    filenames = [pred['filename'] for pred in preds]
    idx = filenames.index(image_id)
    
    pred = preds[idx]
    # gt = gts[idx]
    filename = pred['filename']
    pred_boxes = pred['predictions']
    pred_hoi = pred['hoi_prediction']
    # gt_boxes = gt['annotations']
    # gt_hoi = gt['hoi_annotation']

    img = Image.open(osp.join(img_root, f'HICO_test2015_{filename:08d}.jpg'))
    save_img_i_root = osp.join(save_root, f'test_{filename}')
    if not osp.exists(save_img_i_root):
        os.mkdir(save_img_i_root)

    cat_ids = []
    # for hoi in gt_hoi:
    #     vid = hoi['category_id']
    #     oid = gt_boxes[hoi['object_id']]['category_id']
    #     cat_ids.append((oid, vid))

    for j, hoi in enumerate(pred_hoi):
        vid = hoi['category_id']
        oid = pred_boxes[hoi['object_id']]['category_id']
        # hid = pred_boxes[hoi['subject_id']]['category_id']
        if (oid, vid) in cat_ids:
            # hoi_idx = hois_map.get((oid, vid), 0)
            box_h = pred_boxes[hoi['subject_id']]['bbox']
            box_o = pred_boxes[hoi['object_id']]['bbox']
            score = hoi['score']
            label_o = coco_id_name_map[valid_ids[oid]]
            label_v = vb_map[vid]
            fig, ax = plt.subplots()
            ax.axis('off')
            rect_h = plt.Rectangle((box_h[0], box_h[1]), box_h[2] - box_h[0], box_h[3] - box_h[1], fill=False, color=h_c)
            rect_o = plt.Rectangle((box_o[0], box_o[1]), box_o[2] - box_o[0], box_o[3] - box_o[1], fill=False, color=o_c)
            ax.imshow(img)
            ax.add_patch(rect_h)
            ax.add_patch(rect_o)
                
            fig.savefig(osp.join(save_img_i_root, f'test_{label_o}_{label_v}_{score:.4f}.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
            plt.close(fig)
            
def resize_img_to_same_size(imgs_path):
    n = len(imgs_path)
    for img_path in tqdm(imgs_path, desc='resizing and saving image', total=n):
        img_name = img_path.split('/')[-1].split('.')[0]
        img_dir = osp.join(*img_path.split('/')[:-1])
        img = Image.open(img_path)
        im = img.resize((600, 600))
        im.save(osp.join(img_dir, f'{img_name}_600x600.png'))
        im.close()
        
def preds_with_boxes_vcoco(preds, image_id, img_root='data/CAD_release/frames', save_root='preds_with_boxes'):
    from matplotlib import pyplot as plt
    
    filenames = [pred['filename'] for pred in preds]
    idx = filenames.index(image_id)
    
    pred = preds[idx]
    # gt = gts[idx]
    filename = pred['filename']
    pred_boxes = pred['predictions']
    pred_hoi = pred['hoi_prediction']
    img = Image.open(osp.join(img_root, f'{filename:05d}.jpg'))
    h, w = img.size
    
    save_root = osp.join('logs/video', save_root)
    if not osp.exists(save_root):
        os.mkdir(save_root)
    save_img_i_root = osp.join(save_root, f'test_{filename}')
    if not osp.exists(save_img_i_root):
        os.mkdir(save_img_i_root)
    
    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                             'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                             'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                             'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                             'point_instr', 'read_obj', 'snowboard_instr']
    cat_ids = []
    # for hoi in gt_hoi:
    #     vid = hoi['category_id']
    #     oid = gt_boxes[hoi['object_id']]['category_id']
    #     cat_ids.append((oid, vid))

    for j, hoi in enumerate(pred_hoi):
        vid = hoi['category_id']
        oid = pred_boxes[hoi['object_id']]['category_id']
        # hid = pred_boxes[hoi['subject_id']]['category_id']
        # if (oid, vid) in cat_ids:
        # hoi_idx = hois_map.get((oid, vid), 0)
        box_h = pred_boxes[hoi['subject_id']]['bbox']
        box_o = pred_boxes[hoi['object_id']]['bbox']
        score = hoi['score']
        label_v = verb_classes[vid]
        fig, ax = plt.subplots()
        ax.axis('off')
        rect_h = plt.Rectangle((box_h[0], box_h[1]), box_h[2] - box_h[0], box_h[3] - box_h[1], fill=False, color='r')
        rect_o = plt.Rectangle((box_o[0], box_o[1]), box_o[2] - box_o[0], box_o[3] - box_o[1], fill=False, color='g')
        ax.annotate(label_v, (h // 2, w - 25), xycoords="data", va="center", ha="center", color='w', bbox=dict(boxstyle="square", fill=True, fc="r"))
        ax.imshow(img)
        ax.add_patch(rect_h)
        ax.add_patch(rect_o)
                
        fig.savefig(osp.join(save_img_i_root, f'test_{label_v}_{score:.4f}.png'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        plt.close(fig)
    