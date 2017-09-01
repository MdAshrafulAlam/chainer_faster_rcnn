import numpy as np
import numpy.random as npr
import overlap

class ProposalTargetCreator(object):
    def __init__(self, num_classes):
        self._num_classes = num_classes

    # Proposal ROIs: (0, x1, y1, x2, y2)
    # GT boxes: (x1, y1, x2, y2, label)
    def __call__(self, proposals, gt_boxes):
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        labels, rois, bbox_targets, bbox_inside_weights = _sampled_rois(
                all_rois, gt_boxes, fg_rois_per_image,
                rois_per_image, self._num_classes)

# Bounding box regression target: (class, tx, ty, tw, th)
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    class_name = bbox_target_data[:, 0]
    bbox_targets = np.zeros((class_name.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(class_name > 0)[0]
    for ind in inds:
        cls = class_name[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[inds, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_images,
                 rois_per_image, num_classes):
    overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Foreground ROIs which max_overlaps > FG_THRESH_overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    fg_rois_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_this_image, replace=False)

    # Select background ROIs
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, bg_inds.size)

    # Sample background regions
    if bg_inds.size > 0:
        bg_inds = np.choice(bg_inds, size=bg_rois_this_image, replace=False)

    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled labels
    labels = labels[keep_inds]
    labels[fg_rois_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_target, bbox_inside_weights
