__sets = {}

from pascal_voc import PASCAL_VOC
import numpy as np

for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: PASCAL_VOC(split, year))

def get_imdb(name):
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset')
    return __sets[name]()

def list_imdbs():
    return __sets.keys()
