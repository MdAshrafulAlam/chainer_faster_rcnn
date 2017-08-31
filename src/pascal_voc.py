import os
import numpy as np
import xml.etree.ElementTree as ET
import config

class PASCAL_VOC():
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc' + year + '_' + image_set)
        self._year = year
        self._imageset = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # ROIDB handler
        self._roidb_handler = self.selective_search_roidb #?
        self._salt = str(uuid.uuid4()) #?
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index()

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ground-truth roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ground-truth roidb to {}'.format(cache_file))
        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            gt_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('Loading {}'.format(filename))
        assert os.path.exists(filename), \
                'RPN data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        filename = os.path.join(self._data_path, 'Annotation', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [
                    obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes = cls
            overlaps[ix, cls] = 1.
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

        def _get_comp_id(self):
            comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                    else self._comp_id)
            return comp_id

        def _get_voc_results_file_template(self):
            filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
            path = os.path.join(
                    self._devkit_path,
                    'results',
                    'VOC' + self._year,
                    'Main',
                    filename)
            return path

        def _write_voc_results_file(self, all_boxes):
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                print('Writing {} VOC results file'.format(cls))
                filename = self._get_voc_results_file_template().format(cls)
                with open(filename, 'wt') as f:
                    for im_ind, index in enumerate(self.image_index):
                        dets = all_boxes[cls_ind][im_ind]
                        if dets == []:
                            continue
                        for k in xrange(dets.shape[0]):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, -1],
                                           dets[k, 0] + 1, dets[k, 1] + 1,
                                           dets[k, 2] + 1, dets[k, 3] + 1))

        def _do_python_eval(self, output_dir='output'):
            annopath = os.path.join(self._devkit_path,
                                    'VOC' + self._year,
                                    'Annotations',
                                    '{:s}.xml')
            imagesetfile = os.path.join(self._devkit_path,
                                        'VOC', self._year,
                                        'ImageSets',
                                        'Main',
                                        self._image_set + '.txt')
            cachedir = os.path.join(self._devkit_path, 'annotations_cache')
            aps = []

            use_07_metric = True if int(self._year) < 2010 else False
            print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                rec, prec, ap = voc_eval(filename, annopath,
                                         imagesetfile, cls, cachedir,
                                         ovthresh=0.5,
                                         use_07_metric=use_07_metric)
                aps += [ap]
                print('AP for {} = {:.4f}'.format(cls, ap))
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~~')

        def evaluate_detections(self, all_boxes, output_dir):
            self._write_voc_results_file(all_boxes)
            self._do_python_eval(output_dir)
            if self.config['cleanup']:
                for cls in self._classes:
                    if cls == '__background__':
                        continue
                    filename = self._get_voc_results_file_template().format(cls)
                    os.remove(filename)

if __name__ == '__main__':
    from pascal_voc import PASCAL_VOC
    d = PASCAL_VOC('trainval', '2007')
    res = d.roidb
