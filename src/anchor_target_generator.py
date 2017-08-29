import os
import numpy as np
import numpy.random as npr
from anchor_generator import anchor_generator
import bbox_transform
from overlap import bbox_overlaps

class AnchorTargetCreator(object):
    def __init__(self, feat_stride, allowed_border):
        self.feat_stride = feat_stride
        self.allowed_border = allowed_border

    def __call__(self, rpn_cls_score, gt_boxes, im_info, data):
        anchors = generate_anchors(scales=np.array([8, 16, 32]))
        num_anchors = anchors.shape[0]

        height, width = rpn_cls_score[-2:]
        im_width, im_height = im_info[0, :]

        # Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        A = num_anchors
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # Keep only the anchors inside the image
        inds_inside = np.where(
                (all_anchors[:, 0] >= -self.allowed_border) &
                (all_anchors[:, 1] >= -self.allowed_border) &
                (all_anchors[:, 2] < im_width + self.allowed_border) &
                (all_anchors[:, 3] < im_height + self.allowed_border))[0]
        anchors = all_anchors[inds_inside, :]

        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        # Overlaps between the anchors and the ground-truth boxes
        # TODO: implement overlap calculating method
        overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                 np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # Assign foreground label: for each ground-truth, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # Assign foreground label: above threshold IOU
        labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

        num_fg = int(RPN_FG_FRACTION * FPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dype=np.float32)

        if RPN_POSITIVE_WEIGHT < 0:
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1. / num_examples
            negative_weights = np.ones((1, 4)) * 1. / num_examples
        else:
            assert ((RPN_POSITIVE_WEIGHT > 0) & (RPN_POSITIVE_WEIGHT < 1))
            positive_weights = RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
            negative_weights = (1 - RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))

        bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width

        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_heights.shape[3] == width

def _unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _compute_target(anchors, gt_rois):
    assert anchors.shape[0] == gt_rois.shape[0]
    assert anchors.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(anchors, gt_rois[:, :4]).astype(np.float32, copy=False)

