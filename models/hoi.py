# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sqlalchemy import Identity

import torch
from torch import nn
import torch.nn.functional as F
from models.backbone import build_backbone
from models.matcher import build_cross_matcher, build_matcher
from models.transformer import build_transformer

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.aug_path = 1
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guide_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.obj_class_embed_aug = nn.ModuleList([
            self.obj_class_embed for i in range(self.aug_path)])
        self.verb_class_embed_aug = nn.ModuleList([
            self.verb_class_embed for i in range(self.aug_path)])
        self.sub_bbox_embed_aug = nn.ModuleList([
            self.sub_bbox_embed for i in range(self.aug_path)])
        self.obj_bbox_embed_aug = nn.ModuleList([
            self.obj_bbox_embed for i in range(self.aug_path)])

        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, h_hs, o_hs, inter_hs, _ = self.transformer(
            self.input_proj(
                src), mask, self.query_embed.weight, self.query_embed_h.weight,
            self.query_embed_o.weight, self.pos_guide_embed.weight, pos[-1])

        outputs_obj_class = self.obj_class_embed(hs)
        outputs_verb_class = self.verb_class_embed(hs)
        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()

        outputs_obj_class_aug = torch.stack(
            [self.obj_class_embed_aug[i](h_hs[i]) for i in range(self.aug_path)])
        outputs_verb_class_aug = torch.stack(
            [self.verb_class_embed_aug[i](inter_hs[i]) for i in range(self.aug_path)])
        outputs_sub_coord_aug = torch.stack(
            [self.sub_bbox_embed_aug[i](h_hs[i]).sigmoid() for i in range(self.aug_path)])
        outputs_obj_coord_aug = torch.stack(
            [self.obj_bbox_embed_aug[i](o_hs[i]).sigmoid() for i in range(self.aug_path)])

        if self.training:
            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],
                   'pred_obj_logits_aug': outputs_obj_class_aug[:, -1], 'pred_verb_logits_aug': outputs_verb_class_aug[:, -1],
                   'pred_sub_boxes_aug': outputs_sub_coord_aug[:, -1], 'pred_obj_boxes_aug': outputs_obj_coord_aug[:, -1]}
        else:
            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type, cross_matcher=None, cross_losses=None):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.cross_matcher = cross_matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.cross_losses = cross_losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type

    def jsd_loss(self, p, q):
        m = (p + q) / 2
        jsd = 0.5 * F.kl_div(p.log(), m, reduction='none') + 0.5 * F.kl_div(q.log(), m, reduction='none')
        jsd = jsd.mean(dim=0).mean(dim=-1)
        return jsd
    
    def kl_loss(self, p, q):
        kl = F.kl_div(p.log(), q)
        return kl
    
    def kl_loss_wo_log(self, p, q):
        kl = F.kl_div(p, q)
        return kl

    def loss_obj_labels_jsd(self, outputs, indices, num_interactions):
        out_obj_logits = outputs['pred_obj_logits'].softmax(dim=-1)
        obj_jsd = torch.tensor(0, dtype=torch.float32).to(
            out_obj_logits.device)
        n = len(indices) + 1 # number of path
        for indice in indices:
            # for src, tgt in indice:
            #     n += len(src)
            src_idx = self._get_src_permutation_idx(indice)
            tgt_idx = self._get_tgt_permutation_idx(indice)
            if src_idx[0].shape[0] == 0 or tgt_idx[0].shape[0] == 0:
                continue
            p = out_obj_logits[src_idx]
            q = out_obj_logits[tgt_idx]
            obj_jsd += self.jsd_loss(p, q)
            # obj_jsd = self.kl_loss(p, q)

        # if n != 0:
        #     obj_jsd /= n
        losses = {'obj_jsd': obj_jsd / n}
        return losses

    def loss_verb_labels_jsd(self, outputs, indices, num_interactions):
        out_verb_logits = outputs['pred_verb_logits'].softmax(dim=-1)
        verb_jsd = torch.tensor(0, dtype=torch.float32).to(
            out_verb_logits.device)
        n = len(indices) + 1
        for indice in indices:
            # for src, tgt in indice:
            #     n += len(src)
            src_idx = self._get_src_permutation_idx(indice)
            tgt_idx = self._get_tgt_permutation_idx(indice)
            if src_idx[0].shape[0] == 0 or tgt_idx[0].shape[0] == 0:
                continue
            p = out_verb_logits[src_idx]
            q = out_verb_logits[tgt_idx]
            verb_jsd += self.jsd_loss(p, q)
            # verb_jsd = self.kl_loss(p, q)

        # if n != 0:
        #     verb_jsd /= n
        losses = {'verb_jsd': verb_jsd / n}
        return losses

    def loss_sub_boxes_mse(self, outputs, indices, num_interactions):
        out_sub_boxes = outputs['pred_sub_boxes']
        sub_mse = torch.tensor(0, dtype=torch.float32).to(out_sub_boxes.device)
        n = len(indices) + 1
        for indice in indices:
            # for src, tgt in indice:
            #     n += len(src)
            src_idx = self._get_src_permutation_idx(indice)
            tgt_idx = self._get_tgt_permutation_idx(indice)
            if src_idx[0].shape[0] == 0 or tgt_idx[0].shape[0] == 0:
                continue
            src = out_sub_boxes[src_idx]
            tgt = out_sub_boxes[tgt_idx]
            sub_mse += F.mse_loss(src, tgt, reduce='none') / num_interactions

        # if n != 0:
        #     sub_mse /= n
        losses = {'sub_mse': sub_mse / n}
        return losses

    def loss_obj_boxes_mse(self, outputs, indices, num_interactions):
        out_obj_boxes = outputs['pred_obj_boxes']
        obj_mse = torch.tensor(0, dtype=torch.float32).to(out_obj_boxes.device)
        n = len(indices)
        for indice in indices:
            src_idx = self._get_src_permutation_idx(indice)
            tgt_idx = self._get_tgt_permutation_idx(indice)
            if src_idx[0].shape[0] == 0 and tgt_idx[0].shape[0] == 0:
                continue
            obj_mse += F.mse_loss(out_obj_boxes[src_idx],
                                  out_obj_boxes[tgt_idx], reduce='none') / num_interactions
        losses = {'obj_mse': obj_mse / n}
        return losses

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(
                src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat(
            [t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat(
            [t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(
                src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(
                src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (
                loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (
                loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def get_cross_loss(self, loss, outputs, indices, num, **kwargs):
        loss_map = {
            'obj_jsd': self.loss_obj_labels_jsd,
            'verb_jsd': self.loss_verb_labels_jsd,
            'sub_boxes_mse': self.loss_sub_boxes_mse,
            'obj_boxes_mse': self.loss_obj_boxes_mse
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, indices, num, **kwargs)

    def forward(self, outputs, targets):
        device = outputs['pred_obj_boxes'].device
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs' and 'aug' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        outputs_without_aux_aug = {k[:k.find('_aug')]: v[0] for k, v in outputs.items(
        ) if k != 'aux_outputs' and 'aug' in k}
        indices_aug = self.matcher(outputs_without_aux_aug, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor(
            [num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(
            num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {'loss_obj_aug_ce': torch.tensor(0, dtype=torch.float32).to(device),
                  'loss_verb_aug_ce': torch.tensor(0, dtype=torch.float32).to(device),
                  'loss_sub_aug_ce': torch.tensor(0, dtype=torch.float32).to(device)}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_interactions))
        # [[ augmentation path matcher of original loss]]
        for loss in self.losses:
            k, v = list(self.get_loss(loss, outputs_without_aux_aug, targets,
                        indices_aug, num_interactions).items())[0]  # sum up all original losses
            losses[k] += v

        cross_indices = self.cross_matcher([indices, indices_aug], 2)
        for loss in self.cross_losses:
            _loss = self.get_cross_loss(
                loss, outputs, cross_indices, num_interactions)
            k, v = list(_loss.items())[0]
            if 'obj' in loss:
                losses['loss_obj_aug_ce'] += v
            elif 'sub' in loss:
                losses['loss_sub_aug_ce'] += v
            elif 'verb' in loss:
                losses['loss_verb_aug_ce'] += v
            losses.update(_loss)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
            outputs['pred_verb_logits'], \
            outputs['pred_sub_boxes'], \
            outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack(
            [img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    # vgat = VGAT(args.hidden_dim, args.hidden_dim,
    #             args.num_verb_classes, activation=nn.ReLU())

    if args.hoi:
        model = DETRHOI(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
    matcher = build_matcher(args)
    cross_matcher = build_cross_matcher()
    weight_dict = {}
    if args.hoi:
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
        weight_dict['loss_verb_ce'] = args.verb_loss_coef
        weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_aug_ce'] = args.obj_loss_aug_coef
        weight_dict['loss_verb_aug_ce'] = args.verb_loss_aug_coef
        weight_dict['loss_sub_aug_ce'] = args.sub_loss_aug_coef
    else:
        weight_dict['loss_ce'] = 1
        weight_dict['loss_bbox'] = args.bbox_loss_coef
        weight_dict['loss_giou'] = args.giou_loss_coef
        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.hoi:
        losses = ['obj_labels', 'verb_labels',
                  'sub_obj_boxes', 'obj_cardinality']
        cross_losses = ['obj_jsd', 'verb_jsd',
                        'sub_boxes_mse', 'obj_boxes_mse']
        criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                    weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                    verb_loss_type=args.verb_loss_type, cross_matcher=cross_matcher, cross_losses=cross_losses)
    # else:
    #     losses = ['labels', 'boxes', 'cardinality']
    #     if args.masks:
    #         losses += ["masks"]
    #     criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
        #  eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    if args.hoi:
        postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}
    # else:
    #     postprocessors = {'bbox': PostProcess()}
    #     if args.masks:
    #         postprocessors['segm'] = PostProcessSegm()
    #         if args.dataset_file == "coco_panoptic":
    #             is_thing_map = {i: i <= 90 for i in range(201)}
    #             postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
