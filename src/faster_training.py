import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from anchor_target_generator import AnchorTargetCreator
from proposal_target_generator import ProposalTargetCreator

class FasterRCNNTrainChain(chainer.Chain):
    def __init__(self, faster_rcnn, rpn_sigma=3., roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(16, 0),
                 proposal_target_creator=ProposalTargetCreator(21)):
        super(FasterRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

    def __call__(self, imgs, bboxes, labels, scale):
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        scale = np.asscalar(cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Only batch_size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        print(features.shape)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
                features, img_size, scale)

        # For batch_size 1 only
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois[0]

        # Sampling ROIs and feeding Fast RCNN
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label,
                                                                            self.loc_normalize_mean,
                                                                            self.loc_normalize_std)
        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
        rpn_loc_loss = fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Head losses
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 4)
        roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]
        roi_loc_loss = fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'roi_loc_loss': roi_loc_loss,
                                 'roi_cls_loss': roi_cls_loss,
                                 'loss': loss}, self)
        return loss

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) + (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    loc_loss /= xp.sum(gt_label > 0)
    return loc_loss
