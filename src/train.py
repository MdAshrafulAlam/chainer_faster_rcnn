import argparse
import chainer
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from pascalvoc import VOCDataset
from faster_network import FasterRCNNVGG16
from faster_training import FasterRCNNTrainChain

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

    #TODO
    train_data = VOCDataset(split='trainval', year='2007') #(img, bbox, label, scale)
    test_data = VOCDataset(split='test', year='2007')

    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(num_class),
            pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

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
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log.interval)
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
