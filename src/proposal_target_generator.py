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
        xp = cuda.get_array_module(proposals)
        proposals = cuda.to_cpu(proposals)
        gt_boxes = cuda.to_cpu(gt_boxes)
        labels = cuda.to_cpu(labels)

        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.concatenate((proposals, gt_boxes), axis=0)
        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        rois, bbox_targets, labels = _sample_rois(
                all_rois, gt_boxes, fg_rois_per_image,
                rois_per_image, self._num_classes, labels)
        if xp != np:
            rois = cuda.to_gpu(rois)
            bbox_targets = cuda.to_gpu(bbox_targets)
            labels = cuda.to_gpu(labels)
        return rois, bbox_targets, labels

# Bounding box regression target: (class, tx, ty, tw, th)
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    class_name = bbox_target_data[:, 0]
    bbox_targets = np.zeros((class_name.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(class_name > 0)[0]
    for ind in inds:
        cls = class_name[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[inds, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois):
    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return targets

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes, labels):
    overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = labels[gt_assignment]

    # Foreground ROIs which max_overlaps > FG_THRESH_overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    fg_rois_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_this_image, replace=False)

    # Select background ROIs
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, bg_inds.size)

    # Sample background regions
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_this_image, replace=False)

    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled labels
    labels = labels[keep_inds]
    labels[fg_rois_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_targets = _compute_targets(
            rois, gt_boxes[gt_assignment[keep_inds]])
    # bbox_targets, bbox_inside_weights = \
    #         _get_bbox_regression_labels(bbox_target_data, num_classes)

    return rois, bbox_targets, labels
