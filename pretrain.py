
from pretrainer import PreTrainer
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrain the NN')
    parser.add_argument('--desc', type=str, help='Description for the run (will be appended to saved model and plot')
    parser.add_argument('--dice', action='store_true', help='Use Dice loss')
    parser.add_argument('--usel1', action='store_true', help='Use L1 loss')
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--dilation', type=int, default=1, help='Dilation of encoder')
    parser.add_argument('--no-residual', action='store_true', help='Dont use residual blocks')
    parser.add_argument('--separable', action='store_true', help='Use separable convolution')
    parser.add_argument('--quick_test', action='store_true', help='Test qiuckly by training 2 epochs on the validation set')
    parser.add_argument('--do_all', action='store_true', help='Test qiuckly by training 2 epochs on the validation set')

    args = parser.parse_args()

    desc = args.desc
    usedice = args.dice
    decay = args.decay
    lr = args.lr
    usel1 = args.usel1
    dilation = args.dilation
    if dilation > 2:
        dilation = 2
    if dilation < 1:
        dilation = 1
    no_residual = args.no_residual
    separable = args.separable
    quick_test = args.quick_test
    do_all = args.do_all

    if do_all:
        param_set = [
            [desc, False, False, False, False, 1, lr, decay, quick_test],
            [desc, True, False, False, False, 1, lr, decay, quick_test],
            [desc, True, False, True, False, 1, lr, decay, quick_test],
            [desc, True, False, True, True, 1, lr, decay, quick_test],
            [desc, True, False, True, True, 2, lr, decay, quick_test],
            [desc, True, True, True, True, 2, lr, decay, quick_test],
        ]
        for param in param_set:
            trainer = PreTrainer(*param)
            trainer.run()

    else:
        trainer = PreTrainer(desc, usedice, usel1, not no_residual, separable, dilation, lr, decay, quick_test)
        trainer.run()