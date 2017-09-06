import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from anchor_generator import generate_anchors
from proposal_generator import ProposalCreator

class RegionProposalNetwork(chainer.Chain):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16,
                 initialW=None,
                 proposal_creator_params={},):
        self.anchor_base = generate_anchors()
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(**proposal_creator_params)

        n_anchor = self.anchor_base.shape[0]
        super(RegionProposalNetwork, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, mid_channels, 3, 1, 1, initialW=initialW)
            self.score = L.Convolution2D(mid_channels, n_anchor * 2, 1, 1, 0, initialW=initialW)
            self.loc = L.Convolution2D(mid_channels, n_anchor * 4, 1, 1, 0, initialW=initialW)

    def __call__(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(self.xp.array(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.transpose((0, 2, 3, 1)).reshape((n, -1, 4))

        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.transpose((0, 2, 3, 1))
        rpn_fg_scores = rpn_scores.reshape((n, hh, ww, n_anchor, 2))[:, :, :, :, 1]
        rpn_fg_scores = rpn_fg_scores.reshape((n, -1))
        rpn_scores = rpn_scores.reshape((n, -1, 2))

        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].data, rpn_fg_scores[i].data,
                                      anchor, img_size, scale=scale)
            batch_index = i * self.xp.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = self.xp.concatenate(rois, axis=0)
        roi_indices = self.xp.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    xp = cuda.get_array_module(anchor_base)
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
            shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
