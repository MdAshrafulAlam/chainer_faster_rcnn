from chainer import cuda

def bbox_overlaps(anchors, gt_boxes):
    if anchors.shape[1] != 4 or gt_boxes.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(anchors)

    top_left = xp.maximum(anchors[:, None, :2], gt_boxes[:, :2])
    bottom_right = xp.minimum(anchors[:, None, 2:], gt_boxes[:, 2:])

    area_i = xp.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = xp.prod(anchors[:, 2:] - anchors[:, :2], axis=1)
    area_b = xp.prod(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
