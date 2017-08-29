import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.TRAIN = edict()

__C.TRAIN.SCALES = (600,)

__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of ROIs)
__C.TRAIN.BATCH_SIZE = 256

# Fraction of minibatch that is labeled foreground (class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considerd background (in [0.1, 0.5])
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontal-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and a ground-truth box so that that ROI
# can be used as a bounding box regression in training
__C.TRAIN.BBOX_THRESH = 0.5

# Normalize the targets
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Inside weights
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1., 1., 1., 1.)

__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0., 0., 0., 0.)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before applying NMS
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Min size for proposal's height and width
__C.TRAIN.MIN_SIZE = 16

__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.


#
# Test option
#

__C.TEST = edict()

__C.TEST.SCALES = (600,)

__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for NMS
__C.TEST.NMS = 0.3

__C.TEST.BBOX_REG = True

__C.TEST.HAS_RPN = True

__C.TEST.RPN_NMS_THRESH = 0.7

__C.TEST.RPN_PRE_NMS_TOP_N = 6000

__C.TEST.RPN_POST_NMS_TOP_N = 300

__C.TEST.RPN_MIN_SIZE = 16


#
# MISC
#

__C.RNG_SEED = 3

__C.EPS = 1e-14

__C.GPU_ID = 0

__C.USE_GPU_NMS = True
