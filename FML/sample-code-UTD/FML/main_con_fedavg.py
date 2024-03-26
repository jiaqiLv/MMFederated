from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
from datetime import datetime

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import ASSIGNMENT_CLASS,CLIENT_NUM
from FML_model_guide import MyUTDModelFeature
from FML_design import FeatureConstructor, ConFusionLoss
import data_pre as data

from tqdm import tqdm

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI_personal
from fedml_api.standalone.fedavg.model_trainer import MyModelTrainer
from fedml_api.data_preprocessing.data_loader_default import create_data_loaders
from fedml_api.utils.add_args import add_args
from args import parse_option

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass



def set_loader(opt):
   
    # load data (already normalized)
    num_of_train_unlabel = (opt.num_train_unlabel_basic * (20 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)

    #load labeled train and test data
    x_train_1, x_train_2, y_train = data.load_data(opt.num_class, num_of_train_unlabel, 3, opt.label_rate)
    
    # print('x_train_1.shape:', x_train_1.shape)
    # print('x_train_2.shape:', x_train_2.shape)

    # TODO: temp: make the total number of samples can be evenly divided by the batch size
    x_train_1 = x_train_1[:512]
    x_train_2 = x_train_2[:512]
    y_train = y_train[:512]

    train_dataset = data.Multimodal_dataset(x_train_1, x_train_2, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader


def set_model(opt):
    model = MyUTDModelFeature(input_size=1)
    criterion = ConFusionLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
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

        # print('input_data1.shape:', input_data1.shape)
        # print('input_data2.shape:', input_data2.shape)
        # compute loss
        feature1, feature2 = model(input_data1, input_data2)

        features = FeatureConstructor(feature1, feature2, opt.num_positive)

        loss = criterion(features)

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
    # step1: parameter definition and data loading
    opt = parse_option()
    # parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    # args = parser.parse_args()
    
    num_of_train_unlabel = (opt.num_train_unlabel_basic * (20 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)
    """
    test dataset: 216(27*2*4) in total, except a27_s8_t4 
    x_test_1 (215, 120, 6)
    x_test_2 (215, 40, 20, 3)
    y_test (215,)
    """
    x_test_1, x_test_2, y_test = data.load_data(opt.num_class,num_of_train_unlabel,2,opt.label_rate)

    # step2: trainer
    model_trainer = MyModelTrainer()
    # step3: load data for each client
    train_data_local_dict = dict()
    train_data_local_num_dict = dict()
    test_data_local_dict = dict()
    
    test_dataset = data.Multimodal_dataset(x_test_1,x_test_2,y_test)
    test_data_global = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    
    with open('./class_for_per_client.txt','a') as file:
        formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file.write(str(formatted_time))
        file.write('\n')
          
    # assignment class
    # ASSIGNMENT_CLASS = [
    #     [[18, 24, 15, 22, 20],[0, 23]],
    #     [[13, 6, 14, 22, 1], [24, 3]],
    #     [[8, 15, 3, 25, 14], [11, 26]],
    #     [[20, 21, 10, 3, 17], [5, 15]],
    #     [[12, 6, 18, 3, 9], [4, 8]],
    #     [[2, 16, 14, 13, 1], [17, 25]],
    # ]

    # CLIENT_NUM = 6
    for i in range(CLIENT_NUM):
        print(f'----------{i}--------')
        # num_of_train_unlabel = (opt.num_train_unlabel_basic * (20 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)
        # x_train_1, x_train_2, y_train = data.load_niid_data(opt.num_class, num_of_train_unlabel, 3, opt.label_rate)

        # TODO: modify the data partition mode
        x_train_1, x_train_2, y_train = data.load_niid_data_for_tsne(opt.num_class, 3, opt.label_rate, i, opt, ASSIGNMENT_CLASS)
        assert x_train_1.shape[0] == x_train_2.shape[0] and x_train_1.shape[0] == y_train.shape[0]

        train_dataset_local = data.Multimodal_dataset(x_train_1[:100],x_train_2[:100],y_train[:100])
        train_loader = torch.utils.data.DataLoader(
            train_dataset_local, batch_size=opt.batch_size,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True)
        # train_dataset_labeled_local = data.Multimodal_dataset(x_train_labeled_1,x_train_labeled_2,y_train_labeled)
        # train_labeled_loader = torch.utils.data.DataLoader(
        #     train_dataset_labeled_local, batch_size=opt.batch_size,
        #     num_workers=opt.num_workers, pin_memory=True, shuffle=True)
        test_loader = test_data_global # client and global model use the same test set

        train_data_local_dict[i] = train_loader
        test_data_local_dict[i] = test_loader
        train_data_local_num_dict[i] = x_train_1.shape[0]
        
    dataset = [train_data_local_dict, test_data_global, test_data_local_dict, train_data_local_num_dict]
    opt.client_num_in_total = CLIENT_NUM
    # step4: model group
    global_model, criterion = set_model(opt)
    w_global = model_trainer.get_model_params(global_model)

    local_models = []
    for _ in range(CLIENT_NUM):
        m,_ = set_model(opt)
        model_trainer.set_model_params(m,w_global)
        local_models.append(m)
    # step5: use fedavgAPI for training
    fedavgAPI = FedAvgAPI_personal(dataset, opt.device, opt, model_trainer, global_model, local_models)
    fedavgAPI.train()
    
    # # build data loader
    # train_loader = set_loader(opt)
    # # build model and criterion
    # model, criterion = set_model(opt)
    # # build optimizer
    # optimizer = set_optimizer(opt, model)

    # # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # # training routine
    # for epoch in tqdm(range(1, opt.epochs + 1)):
    #     adjust_learning_rate(opt, optimizer, epoch) # adjust lr for each epoch

    #     # train for one epoch
    #     time1 = time.time()
    #     loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    #     time2 = time.time()
    #     print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

    #     # tensorboard logger
    #     logger.log_value('loss', loss, epoch)
    #     logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    #     if epoch % opt.save_freq == 0:
    #         save_file = os.path.join(
    #             opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    #         save_model(model, optimizer, opt, epoch, save_file)

    # # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)
    # print("num_positive:", opt.num_positive)
    # print("label_rate:", opt.label_rate)
    

if __name__ == '__main__':
    main()
