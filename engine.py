# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from loguru import logger
import numpy as np
import copy
import itertools
import json

import torch
from util.logger import create_small_table, create_table_with_header

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
# from datasets.hico_eval import HICOEvaluator
from datasets.hico_eval_d_ko import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, ramp_w: float = 1):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('ramp_w', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(
            window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(
            window_size=1, fmt='{value:.2f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = {k: v for k,
                       v in criterion.weight_dict.items() if 'aug' not in k}
        weight_dict_cpc = {k: v for k,
                           v in criterion.weight_dict.items() if 'aug' in k}
        # [[ supervision loss for each path ]]
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
        # [[ cpc loss ]]
        losses_cpc = sum(loss_dict[k] * weight_dict_cpc[k]
                         for k in loss_dict.keys() if k in weight_dict_cpc)
        losses_cpc = losses_cpc * ramp_w
        # [[ total loss ]]
        losses += losses_cpc

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error("Loss is {}, stopping training".format(loss_value))
            logger.error(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(
                obj_class_error=loss_dict_reduced['obj_class_error'])
            # metric_logger.update(joint_class_error=loss_dict_reduced['joint_class_error'])
        metric_logger.update(ramp_w=ramp_w)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if utils.get_rank() == 0:
        logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target,
               output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, hoi_save_path=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(
            list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(
            utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    # preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    preds_add_img_id = []
    for i, img_preds in enumerate(preds):
        if i in indices:
            img_preds['img_id'] = torch.tensor(
                [i]).repeat(img_preds['labels'].shape[0])
            preds_add_img_id.append(img_preds)
    preds = preds_add_img_id
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(
            preds, gts, subject_category_id, data_loader.dataset.correct_mat)
    stats = evaluator.evaluate()

    torch.cuda.empty_cache()
    return stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@torch.no_grad()
def evaluate_hoi_video(dataset_file, model, postprocessors, data_loader, subject_category_id, device, hoi_save_path=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(
            list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(
            utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['img_id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    # preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    preds_add_img_id = []
    for i, img_preds in enumerate(preds):
        if i in indices:
            img_preds['img_id'] = torch.tensor(
                [i]).repeat(img_preds['labels'].shape[0])
            preds_add_img_id.append(img_preds)
    preds = preds_add_img_id
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    overlap_iou = 0.5
    max_hois = 100
    out_preds = []
    for i, img_preds in enumerate(preds):
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
        bboxes = [{'bbox': bbox, 'category_id': label} for bbox,
                    label in zip(img_preds['boxes'], img_preds['labels'])]
        hoi_scores = img_preds['verb_scores']
        verb_labels = np.tile(
            np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
        subject_ids = np.tile(
            img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
        object_ids = np.tile(
            img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

        hoi_scores = hoi_scores.ravel()
        verb_labels = verb_labels.ravel()
        subject_ids = subject_ids.ravel()
        object_ids = object_ids.ravel()

        if len(subject_ids) > 0:
            object_labels = np.array(
                [bboxes[object_id]['category_id'] for object_id in object_ids])
            # correct_mat = np.concatenate(
            #     (correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
            # masks = correct_mat[verb_labels, object_labels]
            # hoi_scores *= masks

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
            hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            hois = hois[:max_hois]
        else:
            hois = []

        filename = gts[i]['img_id']
        out_preds.append({
            'filename': filename,
            'predictions': bboxes,
            'hoi_prediction': hois
        })

    with open(hoi_save_path, 'w') as f:
        res = {'preds': out_preds, 'gts': []}
        json.dump(res, f, cls=NumpyEncoder)

    torch.cuda.empty_cache()
    
@torch.no_grad()
def evaluate_kpt_ap(dataset_file, model, postprocessors, data_loader, subject_category_id, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(
            list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(
            utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    # preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    preds_add_img_id = []
    for i, img_preds in enumerate(preds):
        if i in indices:
            img_preds['img_id'] = torch.tensor(
                [i]).repeat(img_preds['labels'].shape[0])
            preds_add_img_id.append(img_preds)
    preds = preds_add_img_id
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)
    elif dataset_file == 'vcoco':
        # evaluator = VCOCOEvaluator(
        #     preds, gts, subject_category_id, data_loader.dataset.correct_mat)
        pass
    # stats = evaluator.evaluate()

    # add keypoint ap
    stats = {}
    if utils.get_rank() == 0:
        out_dir = 'logs/detr-r100-kpt-only'
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        process_kpt(preds, out_dir)
        cocoGt = COCO(os.path.join(
            'data/v-coco/annotations', 'kpt_gts.json'))
        cocoDt = cocoGt.loadRes(os.path.join(out_dir, f'kpt_preds.json'))

        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for ind, name in enumerate(stats_names):
            stats[name] = cocoEval.stats[ind]
        logger.info('==> kpt ap: \n{}'.format(create_table_with_header(stats)))
    torch.cuda.empty_cache()
    return stats


def process_kpt(preds, out_dir):
    kpt = []
    for pred in preds:
        for img_id, bbox, keypoints, scores in zip(pred['img_id'], pred['sub_boxes'], pred['joint_boxes'], pred['joint_scores']):
            data = dict()
            keypoints = torch.cat([keypoints, torch.ones(
                (keypoints.size(-2), 1))], dim=-1).reshape(-1).to(scores.device)
            keypoints = keypoints.reshape(-1)
            data['image_id'] = int(img_id)
            data['keypoints'] = keypoints.tolist()
            data['score'] = float(torch.mean(scores) + torch.max(scores))
            data['category_id'] = 1
            kpt.append(data)

    with open(os.path.join(out_dir, f'kpt_preds.json'), 'w') as f:
        json.dump(kpt, f)
