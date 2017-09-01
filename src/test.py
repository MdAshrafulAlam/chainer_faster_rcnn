from config import cfg
from data_factory import get_imdb
import image_database
import numpy as np
import roidb

def combined_roidb(imdb_name):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset {:s} for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_name.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_name)
    else:
        imdb = get_imdb(imdb_name)
    return imdb, roidb


def get_training_roidb(imdb):
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('Done')

    print('Preparing training data...')
    roidb.prepare_roidb(imdb)
    print('Done')

    return imdb.roidb

def filter_roidb(roidb):
    def is_valid(entry):
        overlaps = entry['max_overlaps']
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                          (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb

if __name__ == '__main__':
    imdb, roidb = combined_roidb('voc_2007_trainval')
    roidb = filter_roidb(roidb)
