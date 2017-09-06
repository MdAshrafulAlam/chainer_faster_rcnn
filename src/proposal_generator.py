import numpy as np
from anchor_generator import generate_anchors
from bbox_transform import bbox_transform_inv, clip_boxes
from config import cfg
import chainer
from chainer import cuda
from nms import non_maximum_suppression

class ProposalCreator(object):
    def __init__(self,):
        self.feat_stride = 16
        self.anchor_scales = np.array([8, 16, 32])
        self.anchors = generate_anchors(scales=self.anchor_scales)
        self.num_anchors = self.anchors.shape[0]

        self.phase = 'TRAIN'
        self.pre_nms_topN = cfg[self.phase].RPN_PRE_NMS_TOP_N
        self.post_nms_topN = cfg[self.phase].RPN_POST_NMS_TOP_N
        self.nms_thresh = cfg[self.phase].RPN_NMS_THRESH
        self.min_size = cfg[self.phase].RPN_MIN_SIZE

    def __call__(self, bbox_deltas, scores,
                 anchors, im_size, scale=1.):
        xp = cuda.get_array_module(bbox_deltas)
        bbox_deltas = cuda.to_cpu(bbox_deltas)
        scores = cuda.to_cpu(scores)
        anchors = cuda.to_cpu(anchors)

        height, width = im_size[0], im_size[1]

        # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        proposals = bbox_transform_inv(anchors, bbox_deltas)
        # proposals = clip_boxes(proposals, im_size)
        proposals[:, slice(0, 4, 2)] = np.clip(
                proposals[:, slice(0, 4, 2)], 0, im_size[0])
        proposals[:, slice(1, 4, 2)] = np.clip(
                proposals[:, slice(1, 4, 2)], 0, im_size[1])

        # Remove predicted boxes with either height or width < threshold
        keep = _filter_boxes(proposals, self.min_size * scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Sort (proposal, scores) by score from highest to lowest
        # Take top pre_nms_topN
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # Apply NMS
        # Take after_nms_topN
        # Return the top proposals
        if xp != np and not self.force_cpu_nms:
            keep = non_maximum_suppression(
                    cuda.to_gpu(proposals), thresh=self.nms_thresh)
            keep = cuda.to_cpu(keep)
        else:
            keep = non_maximum_suppression(
                    proposals, thresh=self.nms_thresh)

        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        proposals = proposals[keep]

        # Output ROIs blob
        # Batch_size = 1 so all batch_inds are 0
        if xp != np:
            proposals = cuda.to_gpu(proposals)
        return proposals

def _filter_boxes(boxes, min_size):
    hs = boxes[:, 2] - boxes[:, 0] + 1
    ws = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
