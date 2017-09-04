import os
from chainer.dataset import download
import utils
from utils import *
import chainer

root = 'data/VOC'
urls = {
        '2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
        'VOCtrainval_11-May-2012.tar',
        '2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
        'VOCtrainval_06-Nov-2007.tar',
        '2007_test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007'
        'VOCtest_06-Nov-2007.tar'
}

def get_voc(year, split):
    if year not in urls:
        raise ValueError
    key = year

    if split == 'test' and year == '2007':
        key = '2007_test'

    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'VOCdevkit/VOC{}'.format(year))
    split_file = os.path.join(base_path, 'ImageSets/Main/{}.txt'.format(split))
    if os.path.exists(split_file):
        return base_path

    download_file_path = utils.cached_download(urls[key])
    ext = os.path.splitext(urls[key])[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path

voc_detection_label_names = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor')

voc_semantic_segmentation_label_names = (('background',) +
                                          voc_detection_label_names)
voc_semantic_segmentation_label_colors = (
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
)

voc_semantic_segmentation_ignore_label_color = (224, 224, 192)

class VOCDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):
        if data_dir == 'auto' and year in ['2007', '2012']:
            data_dir = voc_utils.get_voc(year, split)

        if split not in ['train', 'trainval', 'val']:
            if not (split == 'test' and year == '2007'):
                raise ValueError

        id_list_file = os.path.join(
                data_dir, 'ImageSets/Main/{}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
                os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            for tag in ('ymin', 'xmin', 'ymax', 'xmax'):
                bbox.append(int(bndbox_anno.find(tag).text) - 1)
            name = obj.find('name').text.lower().strip()
            label.append(voc_detection_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)
        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label
