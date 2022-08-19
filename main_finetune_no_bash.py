import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np


from torchvision import datasets
from functions import *
from imagepreprocess import *
from model_init import *
from src.representation import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

# import os
# os.environ["PATH"] += os.pathsep + 'c:/users/chauchatp/anaconda3/lib/site-packages'

class Args(object):
    def __init__(self):
        self.init = 'True'

args = Args()

args.data = 'C:/Users/chauchatp/Documents/Data/CUB_200_2011/CUB_200_2011'
args.arch = 'mpncovresnet50'
args.j = 4
args.epochs = 65
args.start_epoch = 0
args.batch_size = 10
args.lr = 1.2e-3
args.lr_method = 'step'
args.lr_params = [100]
args.momentum = 0.9
args.weight_decay = 1e-3
args.workers = 4
args.print_freq = 100
args.resume = False
args.evaluate = False
args.pretrained = True
args.world_size = 1
args.dist_url = 'tcp://224.66.41.62:23456'
args.dist_backend = 'gloo'
args.seed = None
args.gpe = None
args.modeldir = 'Finetune-CUB-mpncovresnet50-MPNCOV-reproduce-lr0.1-bs10'
# args.representation = 'MPNCOV'
args.representation = 'MPNCOV_SPD'
args.num_classes = 200
args.freezed_layer = 0
args.classifier_factor = 5
args.benchmark = 'CUB'
args.gpu = None





# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

best_prec1 = 0

def main():
    global args, best_prec1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.representation == 'GAvP':
        representation = {'function':GAvP,
                          'input_dim':2048}
    elif args.representation == 'MPNCOV':
        representation = {'function':MPNCOV,
                          'iterNum':5,
                          'is_sqrt':True,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':None if args.pretrained else 256}
    elif args.representation == 'MPNCOV_SPD':
        representation = {'function':MPNCOV_SPD,
                          'iterNum':5,
                          'is_sqrt':True,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':None if args.pretrained else 256}
    elif args.representation == 'BCNN':
        representation = {'function':BCNN,
                          'is_vec':True,
                          'input_dim':2048}
    elif args.representation == 'CBP':
        representation = {'function':CBP,
                          'thresh':1e-8,
                          'projDim':8192,
                          'input_dim': 512}
    else:
        warnings.warn('=> You did not choose a global image representation method!')
        representation = None # which for original vgg or alexnet

    model = get_model(args.arch,
                      representation,
                      args.num_classes,
                      args.freezed_layer,
                      pretrained=args.pretrained)
    # plot network
    # import matplotlib.pyplot as plt
    # plt.plot([1,2,3,4])
    # plt.show()
    # vizNet(model, args.modeldir)
    # obtain learning rate
    LR = Learning_rate_generater(args.lr_method, args.lr_params, args.epochs)
    if args.pretrained:
        params_list = [{'params': model.features.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
        params_list.append({'params': model.representation.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay})
        params_list.append({'params': model.classifier.parameters(),
                            'lr': args.lr*args.classifier_factor,
                            'weight_decay': 0. if args.arch.startswith('vgg') else args.weight_decay})
    else:
        params_list = [{'params': model.features.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
        params_list.append({'params': model.representation.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay})
        params_list.append({'params': model.classifier.parameters(),
                            'lr': args.lr*args.classifier_factor,
                            'weight_decay':args.weight_decay})

    optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(args.benchmark)

    train_dataset = datasets.ImageFolder(
        traindir,
        train_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ## init evaluation data loader
    if evaluate_transforms is not None:
        evaluate_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, evaluate_transforms),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if evaluate_transforms is not None:
            validate(evaluate_loader, model, criterion)
        validate(val_loader, model, criterion)
        return
    # make directory for storing checkpoint files
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, LR.lr_factor, epoch)
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion)
        # update stats
        stats_._update(trainObj, top1, top5, valObj, prec1, prec5)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = []
        if args.store_model_everyepoch:
            filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
        else:
            filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        plot_curve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
    if evaluate_transforms is not None:
        model_file = os.path.join(args.modeldir, 'model_best.pth.tar')
        print("=> loading best model '{}'".format(model_file))
        print("=> start evaluation")
        best_model = torch.load(model_file)
        model.load_state_dict(best_model['state_dict'])
        validate(evaluate_loader, model, criterion)




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        input.cuda()
        print(input.device)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            ## modified by jiangtao xie
            if len(input.size()) > 4:# 5-D tensor
                bs, crops, ch, h, w = input.size()
                output = model(input.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Learning_rate_generater(object):
    """Generates a list of learning rate for each training epoch"""
    def __init__(self, method, params, total_epoch):
        if method == 'step':
            print(params)
            print(total_epoch)
            lr_factor, lr = self.step(params, total_epoch)
        elif method == 'log':
            lr_factor, lr = self.log(params, total_epoch)
        else:
            raise KeyError("=> undefined learning rate method '{}'" .format(method))
        self.lr_factor = lr_factor
        self.lr = lr
    def step(self, params, total_epoch):
        decrease_until = params
        decrease_num = len(decrease_until)
        base_factor = 0.1
        lr_factor = torch.ones(total_epoch, dtype=torch.double)
        lr = [args.lr]
        for num in range(decrease_num):
            if decrease_until[num] < total_epoch:
                lr_factor[int(decrease_until[num])] = base_factor
        for epoch in range(1,total_epoch):
            lr.append(lr[-1]*lr_factor[epoch])
        return lr_factor, lr
    def log(self, params, total_epoch):
        params = params[0]
        left_range = params[0]
        right_range = params[1]
        np_lr = np.logspace(left_range, right_range, total_epoch)
        lr_factor = [1]
        lr = [np_lr[0]]
        for epoch in range(1, total_epoch):
            lr.append(np_lr[epoch])
            lr_factor.append(np_lr[epoch]/np_lr[epoch-1])
        if lr[0] != args.lr:
            args.lr = lr[0]
        return lr_factor, lr


def adjust_learning_rate(optimizer, lr_factor, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    groups = ['features']
    groups.append('representation')
    groups.append('classifier')
    num_group = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_factor[epoch]
        print('the learning rate is set to {0:.5f} in {1:} part'.format(param_group['lr'], groups[num_group]))
        num_group += 1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
