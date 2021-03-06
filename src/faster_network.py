import collections
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from vgg16 import VGG16

from faster_rcnn import FasterRCNN
from region_proposal_network import RegionProposalNetwork

from utils.download import *

class FasterRCNNVGG16(FasterRCNN):
    _models = {
        'voc07': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.4/'
            'faster_rcnn_vgg16_voc07_trained_2017_08_06.npz'
        }
    } # Add COCO or VOC12 here
    feat_stride = 16

    def __init__(self,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 vgg_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params={}):
        if n_fg_class is None:
            if pretrained_model not in self._models:
                raise ValueError('The n_fg_class needs to be supplied as an argument')
            n_fg_class = self._models[pretrained_model]['n_fg_class']

        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if vgg_initialW is None and pretrained_model:
            vgg_initialW = chainer.initializers.constant.Zero()

        extractor = VGG16(initialW=vgg_initialW)
        extractor.feature_names = 'conv5_3'
        extractor.remove_unused()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = VGG16ROIHead(
            n_fg_class + 1,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            vgg_initialW=vgg_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
            mean=np.array([122.7717, 115.9465, 102.9801],
                          dtype=np.float32)[:, None, None],
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_vgg16()
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _copy_imagenet_pretrained_vgg16(self):
        pretrained_model = VGG16(pretrained_model='imagenet')
        self.extractor.conv1_1.copyparams(pretrained_model.conv1_1)
        self.extractor.conv1_2.copyparams(pretrained_model.conv1_2)
        self.extractor.conv2_1.copyparams(pretrained_model.conv2_1)
        self.extractor.conv2_2.copyparams(pretrained_model.conv2_2)
        self.extractor.conv3_1.copyparams(pretrained_model.conv3_1)
        self.extractor.conv3_2.copyparams(pretrained_model.conv3_2)
        self.extractor.conv3_3.copyparams(pretrained_model.conv3_3)
        self.extractor.conv4_1.copyparams(pretrained_model.conv4_1)
        self.extractor.conv4_2.copyparams(pretrained_model.conv4_2)
        self.extractor.conv4_3.copyparams(pretrained_model.conv4_3)
        self.extractor.conv5_1.copyparams(pretrained_model.conv5_1)
        self.extractor.conv5_2.copyparams(pretrained_model.conv5_2)
        self.extractor.conv5_3.copyparams(pretrained_model.conv5_3)
        self.head.fc6.copyparams(pretrained_model.fc6)
        self.head.fc7.copyparams(pretrained_model.fc7)

class VGG16ROIHead(chainer.Chain):
    def __init__(self, n_class, roi_size, spatial_scale,
                 vgg_initialW=None, loc_initialW=None, score_initialW=None):
        super(VGG16ROIHead, self).__init__()
        with self.init_scope():
            self.fc6 = L.Linear(25088, 4096, initialW=vgg_initialW)
            self.fc7 = L.Linear(4096, 4096, initialW=vgg_initialW)
            self.cls_loc = L.Linear(4096, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(4096, n_class, initialW=score_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate((roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(x, indices_and_rois,
                                 self.roi_size, self.roi_size,
                                 self.spatial_scale)
        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
