import argparse
import matplotlib.pyplot as plt
import chainer
from pascalvoc import voc_detection_label_names, voc_semantic_segmentation_label_colors
from faster_network import FasterRCNNVGG16
from utils.image import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc07')
    parser.add_argument('image')
    args = parser.parse_args()

    model = FasterRCNNVGG16(
            n_fg_class=len(voc_detection_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    print(bbox)
    print(label)
    print(score)

    draw_bbox(img, bbox, label, score, label_names=voc_detection_label_names)
    plt.savefig('result.jpg')

def draw_bbox(img, bboxes, labels=None, scores=None, label_names=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))

    if len(bboxes) == 0:
        return ax

    for i, bbox in enumerate(bboxes):
        xy = (bbox[1], bbox[0])
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=np.array(voc_semantic_segmentation_label_colors[labels[i] + 1]) / 255., linewidth=3))

        caption = list()
        if labels is not None and label_names is not None:
            label = labels[i]
            if not (label >= 0 and label < len(label_names)):
                raise ValueError
            caption.append(label_names[label])
        if scores is not None:
            score = scores[i]
            caption.append('{:.2f}'.format(score))

        if len(caption) > 0:
            ax.text(bbox[1], bbox[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax

if __name__ == '__main__':
    main()
