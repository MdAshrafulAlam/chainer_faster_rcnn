import os
import os.path as osp
import PIL
import overlap
import numpy as np
import scipy.sparse
from config import cfg

class ImageDatabase(object):
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        # self._roidb_handler = self.default_roidb
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    # @property
    # def roidb_handler(self):
    #     return self._roidb_handler

    # @roidb_handler.setter
    # def roidb_handler(self, val):
    #     self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dicts, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        print('roidb is None')
        # self._roidb = self.roidb_handler()
        # return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        # all_boxes[class][image] = [] or np.array of shape #dets x 5
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in xrange(self.num_images)]

    def _append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128:256': 5, '256:512': 6, '512-inf': 7}
        area_ranges = [[0**2, 1e5**2],   # all
                       [0**2, 32**2],    # small
                       [32**2, 96**2],   # medium
                       [96**2, 1e5**2],  # large
                       [96**2, 128**2],  # 96-128
                       [128**2, 256**2], # 128-256
                       [256**2, 512**2], # 256-512
                       [512**2, 1e5**2], # 512-inf
                      ]
        assert areas.has_key(area), 'Unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in xrange(self.num_images):
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                # Find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)

                # and get the IOU amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)

                # Find which gt box is best covered (most IOU)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert gt_ovr >= 0

                # Find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]

                # Record the IOU coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr

                # Mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.5
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)

        # Compute recall for each IOU threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            print(boxes)
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                                gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidb(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

