"""
from __future__ import print_function
import os
from datetime import datetime
import numpy as np
import torch
import dataloaders.cifar100 as dataloader
from networks.arch_craft import Net
from approaches import sgd as approach
import copy

Inc_cls = 5    # Number of classes incremented at each step

#generated_code
class TrainModel(object):
    def __init__(self):
        self.grad_clip = 10
        self.epoch = 20
        self.lr = 0.01
        self.code = code
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.inc = Inc_cls

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode, encoding='utf-8')
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def process(self, s, data_lock):
        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])
        with data_lock:
            data, taskcla, _ = dataloader.get(seed=s, inc=self.inc)
        approach.set_random(s)
        net = Net(taskcla, depth, width, pool_code, double_code)
        net = net.cuda()
        total = sum([param.nelement() for param in net.parameters()])
        self.log_record('Number of parameter: %.4fM' % (total / 1e6))

        appr = approach.Appr(net, nepochs=self.epoch, sbatch=128, lr=self.lr, clipgrad=self.grad_clip, seed=s)
        self.log_record('-' * 100)

        # Loop tasks
        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        aps = []
        afs = []
 
        for t, ncla in taskcla:
            self.log_record('*' * 100)
            self.log_record('Task {:2d} ({:s})'.format(t, data[t]['name']))
            self.log_record('*' * 100)

            # Get data
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

            # Train
            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            self.log_record('-' * 100)

            # Test
            for u in range(t + 1):
                xtest = data[u]['test']['x'].cuda()
                ytest = data[u]['test']['y'].cuda()
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                self.log_record('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'],
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
            self.log_record('ap:%.5f' % now_acc)

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
                self.log_record('af:%.5f' % af)

        # Done
        self.log_record('*' * 100)

        self.log_record(str(aps))
        self.log_record(str(afs))
        aia = 0.0
        for ap in aps:
            aia += ap
        aia /= acc.shape[1]
        aia *= 100
        self.log_record('aia:%.3f' % aia)
        final_acc = 0.0
        for k in range(acc.shape[1]):
            final_acc += float(acc[-1, k])
        final_acc /= acc.shape[1]
        final_acc *= 100
        af = afs[-1] * 100
        self.log_record('aa:%.3f' % final_acc)
        self.log_record('af:%.3f' % af)
        self.log_record('Done!')
        return aia

class RunModel:
    def do_work(self, gpu_id, file_id, data_lock):
        torch.cuda.set_device(gpu_id)
        model = TrainModel()
        model.log_record(f'Using CUDA GPU {gpu_id}', first_time=True)
        scores = []
        for seed in approach.DEFAULT_SEEDS:
            model.log_record(f'Seed: {seed}')
            scores.append(model.process(seed, data_lock))
        mean_aia = float(np.mean(scores))
        model.log_record(f'AIA over {len(scores)} runs: {mean_aia:.3f} ± {np.std(scores):.3f}%')
        return mean_aia
"""
