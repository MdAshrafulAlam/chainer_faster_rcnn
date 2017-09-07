import numpy as np
import numpy.random as npr
from overlap import bbox_overlaps
from config import cfg
from bbox_transform import bbox_transform
from chainer import cuda

class ProposalTargetCreator(object):
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def __call__(self, proposals, gt_boxes, labels,
                 loc_normalize_means=(0., 0., 0., 0.),
                 loc_normalize_stds=(0.1, 0.1, 0.2, 0.2)):
        proposals = cuda.to_cpu(proposals)
        gt_boxes = cuda.to_cpu(gt_boxes)
        labels = cuda.to_cpu(labels)

        all_rois = np.concatenate((proposals, gt_boxes), axis=0)
        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        rois, bbox_targets, gt_labels = _sample_rois(
                all_rois, gt_boxes, fg_rois_per_image,
                rois_per_image, self._num_classes, labels)
        return rois, bbox_targets, gt_labels

def _compute_targets(ex_rois, gt_rois):
    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, np.float32))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, np.float32))
    return targets

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes, labels):
    overlaps = bbox_overlaps(all_rois, gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    gt_labels = labels[gt_assignment] + 1

    # Foreground ROIs which max_overlaps > FG_THRESH_overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    fg_rois_this_image = int(min(fg_rois_per_image, fg_inds.size))

    # Sample foreground regions
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_this_image, replace=False)

    # Select background ROIs
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = int(min(bg_rois_this_image, bg_inds.size))

    # Sample background regions
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_this_image, replace=False)

    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled labels
    gt_labels = gt_labels[keep_inds]
    gt_labels[fg_rois_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_targets = _compute_targets(
            rois, gt_boxes[gt_assignment[keep_inds]])

    xp = cuda.get_array_module(rois)
    if xp != np:
        rois = cuda.to_gpu(rois)
        bbox_targets = cuda.to_gpu(bbox_targets)
        gt_labels = cuda.to_gpu(gt_labels)

    return rois, bbox_targets, gt_labels
