import numpy as np


# Example ROIs: (xa_1, ya_1, xa_2, ya_2)
# Ground-truth ROIS: (x*_1, y*_1, x*_2, y*_2)
# Return (t*_x, t*_y, t*_w, t*_h)
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(anchors, deltas):
    if anchors.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.type)

    anchors = anchors.astype(delta.type, copy=False)

    anchor_widths = anchors[:, 2] - anchors[:, 0] + 1.
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * anchor_widths[:, np.newaxis] + anchor_ctr_x[:, np.newaxis]
    pred_ctr_y = dy * anchor_heights[:, np.newaxis] + anchor_ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * anchor_widths[:, np.newaxis]
    pred_h = np.exp(dh) * anchor_heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
