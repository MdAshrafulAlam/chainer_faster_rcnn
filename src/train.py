import argparse
import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from pascalvoc import VOCDataset, voc_detection_label_names
from faster_network import FasterRCNNVGG16
from faster_training import FasterRCNNTrainChain
import numpy as np
from bbox_transform import resize_bbox, flip_bbox, random_flip

class Transform(object):
    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        img, params = random_flip(
                img, x_random=True, return_param=True)
        bbox = flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale

def main():
    parser = argparse.ArgumentParser(
            description='Faster R-CNN Chainer version')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='result',
            help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data = VOCDataset(split='trainval', year='2007')
    # test_data = VOCDataset(split='test', year='2007',
    #                        use_difficult=True, return_difficult=True)

    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(voc_detection_label_names),
            pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, Transform(faster_rcnn))

    train_iter = chainer.iterators.MultiprocessIterator(
            train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    # test_iter = chainer.iterators.SerialIterator(
    #         test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(
            updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
            extensions.snapshot_object(model.faster_rcnn, 'snapshot_model.npz'),
            trigger=(args.iteration, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 5, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
        ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
                           ['main/loss'],
                           file_name='loss.png', trigger=plot_interval
                      ),
                      trigger=plot_interval)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.run()

if __name__ == '__main__':
    main()
