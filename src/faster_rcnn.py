import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

class FasterRCNN(chainer.Chain):
    def __init__(self, extractor, rpn, head, mean,
                 min_size=600,
                 max_size=1000,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),):
        super().__init__()

        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.use_preset('visualize') # not sure what this is

    @property
    def n_class(self):
        return head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor =\
                self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('undefined preset')

    def prepare(self, img):
        _, H, W = img.shape
        scale = 1.
        scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape(-1, self.n_class, 4)[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def predict(self, imgs):
        prepared_imgs = list()
        scales = list()
        for img in imgs:
            _, H, W = img.shape
            img = self.prepare(img.astype(np.float32))
            scale = img.shape[2] / W
            prepared_imgs.append(img)
            scales.append(scale)

        bboxes = list()
        labels = list()
        scores = list()
        for img, scale in zip(prepared_imgs, scales):
            with chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                H, W = img_var.shape[2:]
                roi_cls_locs, roi_scores, rois, _ = self.__call__(img_var, scale=scale)
            # Assuming that batch size is 1
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi = rois / scale

            # Converting predictions to bounding boxes
            # Bounding boxes are scaled to the scale of input images
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean), self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std), self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 4)
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
            cls_bbox = loc2bbox(roi.reshape(-1, 4), roi_cls_loc.reshape(-1, 4))
            cls_bbox = cls_bbox.reshape(-1, self.n_class * 4)

            prob = F.softmax(roi_score).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_prob = cuda.to_cpu(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
