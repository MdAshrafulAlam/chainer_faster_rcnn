import os
import numpy as np
import numpy.random as npr
from anchor_generator import generate_anchors
from bbox_transform import bbox_transform
from overlap import bbox_overlaps
from config import cfg
from chainer import cuda

class AnchorTargetCreator(object):
    def __init__(self, feat_stride, allowed_border):
        self.feat_stride = feat_stride
        self.allowed_border = allowed_border

    def __call__(self, gt_boxes, anchors, im_size):
        xp = cuda.get_array_module(gt_boxes)
        gt_boxes = cuda.to_cpu(gt_boxes)
        anchors = cuda.to_cpu(anchors)

        num_anchors = anchors.shape[0]

        im_height, im_width = im_size

        # Keep only the anchors inside the image
        inds_inside = xp.where(
                (anchors[:, 0] >= -self.allowed_border) &
                (anchors[:, 1] >= -self.allowed_border) &
                (anchors[:, 2] < im_height + self.allowed_border) &
                (anchors[:, 3] < im_width + self.allowed_border))[0]
        anchors = anchors[inds_inside, :]

        labels = np.empty((len(inds_inside),), dtype=np.int32)
        labels.fill(-1)

        # Overlaps between the anchors and the ground-truth boxes
        # TODO: implement overlap calculating method
        overlaps = bbox_overlaps(anchors, gt_boxes)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # Assign foreground label: for each ground-truth, anchor with highest overlap
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        labels[gt_argmax_overlaps] = 1

        # Assign foreground label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps])
        '''
        bbox_targets = xp.zeros((len(inds_inside), 4), dtype=xp.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = xp.zeros((len(inds_inside), 4), dtype=xp.float32)
        bbox_inside_weights[labels == 1, :] = xp.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = xp.zeros((len(inds_inside), 4), dtype=xp.float32)

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = xp.sum(labels >= 0)
            positive_weights = xp.ones((1, 4)) * 1. / num_examples
            negative_weights = xp.ones((1, 4)) * 1. / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / xp.sum(labels == 1)
            negative_weights = (1 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / xp.sum(labels == 0)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
        '''

        labels = _unmap(labels, num_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, num_anchors, inds_inside, fill=0)
        # bbox_inside_weights = _unmap(bbox_inside_weights, num_anchors, inds_inside, fill=0)
        # bbox_outside_weights = _unmap(bbox_outside_weights, num_anchors, inds_inside, fill=0)

        if xp != np:
            bbox_targets = cuda.to_gpu(bbox_targets)
            labels = cuda.to_gpu(labels)
        return bbox_targets, labels

def _unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_targets(anchors, gt_rois):
    return bbox_transform(anchors, gt_rois[:, :4]).astype(np.float32, copy=False)
