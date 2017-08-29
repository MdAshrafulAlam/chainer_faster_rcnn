import numpy as np
from anchor_generator import generate_anchors
from bbox_transform import bbox_transform_inv, clip_boxes

class ProposalGenerator(object):
    def __init__(self,):
        self.feat_stride = 16
        self.anchor_scales = np.array([8, 16, 32])
        self.anchors = generate_anchors(scales=anchor_scales))
        self.num_anchors = anchors.shape[0]

        self.phase = PHASE
        self.pre_nms_topN = cfg[self.phase].RPN_PRE_NMS_TOP_N
        self.post_nms_topN = cfg[self.phase].RPN_POST_NMS_TOP_N
        self.nms_thresh = cfg[self.phase].RPN_NMS_THRESH
        self.min_size = cfg[self.phase].RPN_MIN_SIZE

    def __call__(self, rpn_cls_prob, im_info):
        scores = rpn_cls_prob[:, self.num_anchors, :, :]
        bbox_deltas = rpn_bbox_pred
        im_info = im_info[0, :]

        height, width = scores[-2:]

        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        A = self.num_anchors
        K = shifts.shape[0]
        anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        proposals = clip_boxes(proposals, im_info[:2])

        # Remove predicted boxes with either height or width < threshold
        keep = _filter_boxes(proposals, self.min_size * im_info[2])
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
        keep = nms(np.hstack((proposals, scores)), self.nms_thresh)
        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output ROIs blob
        # Batch_size = 1 so all batch_inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return blob

def _filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
