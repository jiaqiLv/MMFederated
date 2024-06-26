from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from FML_model_guide import MyUTDModelFeature
from FML_design import FeatureConstructor, ConFusionLoss
from focal_loss.focal_loss import FOCALLoss
import data_pre as data

from tqdm import tqdm

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI_personal
from fedml_api.standalone.fedavg.model_trainer import MyModelTrainer

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from params.train_params import parse_train_params
args = parse_train_params()

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200,300',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='UTD-MHAD',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')
    parser.add_argument('--num_train_basic', type=int, default=1,
                        help='num_train_basic')
    parser.add_argument('--num_train_unlabel_basic', type=int, default=1,
                        help='num_train_unlabel_basic')
    parser.add_argument('--label_rate', type=int, default=10,
                        help='label_rate')

    # method
    parser.add_argument('--method', type=str, default='FML',
                        choices=['FML'], help='choose method')
    parser.add_argument('--num_positive', type=int, default=9,
                        help='num_positive')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=int, default=1,
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/FML/{}_models'.format(opt.dataset)
    opt.tb_path = './save/FML/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_label_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_epoch_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.label_rate, opt.learning_rate,
               opt.lr_decay_rate, opt.batch_size, opt.temp, opt.trial, opt.epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
   
    # load data (already normalized)
    num_of_train_unlabel = (opt.num_train_unlabel_basic * (20 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)

    #load labeled train and test data
    x_train_1, x_train_2, y_train = data.load_data(opt.num_class, num_of_train_unlabel, 3, opt.label_rate)
    # print('x_train_1.shape:', x_train_1.shape)
    # print('x_train_2.shape:', x_train_2.shape)

    train_dataset = data.Multimodal_dataset(x_train_1, x_train_2, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):
    model = MyUTDModelFeature(input_size=1)
    # criterion = ConFusionLoss(temperature=opt.temp)
    criterion = FOCALLoss(args=args)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input_data1, input_data2, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
        bsz = input_data1.shape[0]

        # compute loss
        feature1, feature2 = model(input_data1, input_data2)

        # features = FeatureConstructor(feature1, feature2, opt.num_positive)
        # loss = criterion(features)

        # print('feature1.shape:', feature1.shape)
        # print('feature2.shape:', feature2.shape)
        loss = criterion(feature1,feature2)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()
    # build data loader
    train_loader = set_loader(opt)
    # build model and criterion
    model, criterion = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in tqdm(range(1, opt.epochs + 1)):
        adjust_learning_rate(opt, optimizer, epoch) # adjust lr for each epoch

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, opt, epoch, save_file)
            # save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
