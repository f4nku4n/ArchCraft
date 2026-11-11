from __future__ import print_function
import argparse
import os
import numpy as np
from dataloaders import cifar100, imagenet100
from approaches import sgd as approach
import utils
import copy

ARCHITECTURES = {
    'alexac-a': [2, 40, [1, 1, 2, 2, 2], [0, 0, 0, 1, 2]],
    'alexac-b': [2, 32, [1, 1, 2, 2, 2], [0, 1, 2, 2, 2]],
    'alexnet': None,
}

def run_test():
    parser = argparse.ArgumentParser(description='Evaluate Task IL on CIFAR-100 or ImageNet-100.')
    parser.add_argument('--dataset', choices=['cifar100', 'imagenet100'], default='cifar100')
    parser.add_argument('--data-root', help='ImageNet-100 folder containing train/ and val/')
    parser.add_argument('--increment', type=int, choices=[5, 10], default=5)
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='alexac-a')
    parser.add_argument('--seeds', type=int, nargs='+', default=approach.DEFAULT_SEEDS)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.dataset == 'imagenet100' and not args.data_root:
        parser.error('--data-root is required for ImageNet-100')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = TrainModel(ARCHITECTURES[args.architecture], args.increment, args.dataset, args.data_root)
    results = []
    for seed in args.seeds:
        la, aia = model.process(seed)
        results.append((la, aia))
        print(f'Seed {seed}: LA={la:.3f}%, AIA={aia:.3f}%')

    for metric, values in zip(('LA', 'AIA'), np.asarray(results).T):
        print(f'{metric} over {len(values)} runs: {np.mean(values):.3f} ± {np.std(values):.3f}%')

class TrainModel(object):
    def __init__(self, code, increment, dataset, data_root):
        self.grad_clip = 10
        self.epoch = 20
        self.lr = 0.01
        self.code = code
        self.inc = increment
        self.dataset = dataset
        self.data_root = data_root

    def process(self, s):
        print('\n\n')
        print(f'Dataset: {self.dataset}, increment: {self.inc}, seed: {s}')
        if self.dataset == 'imagenet100':
            data, taskcla, _ = imagenet100.get(seed=s, inc=self.inc, data_root=self.data_root)
        else:
            data, taskcla, _ = cifar100.get(seed=s, inc=self.inc)

        approach.set_random(s)
        if self.code is None:
            from networks.alexnet import Net
            net = Net(taskcla)
        else:
            from networks.arch_craft import Net
            net = Net(taskcla, *copy.deepcopy(self.code))

        net = net.cuda()
        total = sum([param.nelement() for param in net.parameters()])
        print('Number of parameter: %.4fM' % (total / 1e6))
        appr = approach.Appr(net, nepochs=self.epoch, sbatch=128, lr=self.lr, clipgrad=self.grad_clip, seed=s)
        print(appr.criterion)
        utils.print_optimizer_config(appr.optimizer)
        print('-' * 100)

        # Loop tasks
        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        aps = []
        afs = []
 
        for t, ncla in taskcla:
            print('*' * 100)
            print('Task {:2d} ({:s})'.format(t, data[t]['name']))
            print('*' * 100)

            # Get data
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

            # Train
            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            print('-' * 100)

            # Test
            for u in range(t + 1):
                xtest = data[u]['test']['x'].cuda()
                ytest = data[u]['test']['y'].cuda()
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'],
                                                                                              test_loss,
                                                                                              100 * test_acc))
                acc[t, u] = test_acc
                lss[t, u] = test_loss
                
            now_acc = 0.0
            for k in range(t+1):
                now_acc += float(acc[t, k])
            now_acc /= (t+1)
            round(now_acc, 5)
            aps.append(now_acc)
            print('ap:%.5f' % now_acc)

            if t != 0:
                f = 0.0
                for k in range(t):
                    max_acc = 0.0
                    for j in range(k, t):
                        if acc[j, k] > max_acc:
                            max_acc = acc[j, k]
                    f += float(max_acc - acc[t, k])
                af = f/t
                round(af, 5)
                afs.append(af)
                print('af:%.5f' % af)

        # Done
        print('*' * 100)
        print('Accuracies =')
        for i in range(acc.shape[0]):
            print('\t', end='')
            for j in range(acc.shape[1]):
                print('{:5.1f}% '.format(100 * acc[i, j]), end='')
            print()
        print('*' * 100)
        print(aps)
        aia = 0.0
        for ap in aps:
            aia += ap
        aia /= len(taskcla)
        print('aia:%.5f' % aia)
        print(afs)
        final_acc = 0.0
        for k in range(acc.shape[1]):
            final_acc += float(acc[-1, k])
        final_acc /= acc.shape[1]
        round(final_acc, 5)
        print('final_acc:%.5f' % final_acc)
        print('Done!')
        return final_acc * 100, aia * 100

if __name__ == '__main__':
    run_test()
