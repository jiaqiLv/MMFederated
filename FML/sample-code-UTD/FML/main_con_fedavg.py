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
from fedml_api.standalone.fedavg.model_trainer_fedprox import FedProxTrainer
from fedml_api.data_preprocessing.data_loader_default import create_data_loaders
from fedml_api.utils.add_args import add_args
from args import parse_option

import logging

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

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

def main():
    formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not os.path.exists(f'model/{formatted_time}'):
        os.mkdir(f'model/{formatted_time}')
    logging.basicConfig(filename=f'model/{formatted_time}/log.log',level=logging.INFO)
    # step1: parameter definition
    opt = parse_option()
    opt.formatted_time = formatted_time
    # step2: test dataset loading
    num_of_train_unlabel = (opt.num_train_unlabel_basic * (20 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)
    """
    test dataset: 216(27*2*4) in total, except a27_s8_t4 
    x_test_1 (215, 120, 6)
    x_test_2 (215, 40, 20, 3)
    y_test (215,)
    """
    x_test_1, x_test_2, y_test = data.load_data(opt.num_class,num_of_train_unlabel,2,opt.label_rate)
    test_dataset = data.Multimodal_dataset(x_test_1,x_test_2,y_test)
    test_data_global = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    # step3: trainer
    # model_trainer = MyModelTrainer()
    model_trainer = FedProxTrainer()
    # step4: load training data for each client
    train_data_local_dict = dict()
    train_data_local_num_dict = dict()
    test_data_local_dict = dict()
    train_label_data_local_dict = dict()
    
    with open('./class_for_per_client.txt','a') as file:
        formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file.write(str(formatted_time))
        file.write('\n')
          
    for i in range(CLIENT_NUM):
        print(f'----------{i}--------')
        """
        Explanation:
        Both labeled and unlabeled data have the same category, 5 for major class and 2 for minor class, for example:
        num_of_major_class: [18, 24, 15, 22, 20]
        num_of_minor_class: [0, 23]
        unlabeled data: 5*19+2*2=99(without file missing)
        labeled data: depends on the labeled data percent
        """
        x_train_1, x_train_2, y_train = data.load_niid_data_for_tsne(opt.num_class, 3, opt.label_rate, i, opt, ASSIGNMENT_CLASS)
        x_train_labeled_1, x_train_labeled_2, y_train_labeled = data.load_niid_data_for_tsne(opt.num_class,1,opt.label_rate,i,opt,ASSIGNMENT_CLASS)
        assert x_train_1.shape[0] == x_train_2.shape[0] and x_train_1.shape[0] == y_train.shape[0]
        assert x_train_labeled_1.shape[0] == x_train_labeled_2.shape[0] and x_train_labeled_1.shape[0] == y_train_labeled.shape[0]

        train_dataset_local = data.Multimodal_dataset(x_train_1,x_train_2,y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_local, batch_size=opt.batch_size,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True)
        train_dataset_labeled_local = data.Multimodal_dataset(x_train_labeled_1,x_train_labeled_2,y_train_labeled)
        train_labeled_loader = torch.utils.data.DataLoader(
            train_dataset_labeled_local, batch_size=opt.batch_size,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True)
        test_loader = test_data_global # client and global model use the same test set

        train_data_local_dict[i] = train_loader
        test_data_local_dict[i] = test_loader
        train_data_local_num_dict[i] = x_train_1.shape[0]
        train_label_data_local_dict[i] = train_labeled_loader
        
    dataset = [train_data_local_dict, test_data_global, test_data_local_dict, train_data_local_num_dict]
    opt.client_num_in_total = CLIENT_NUM
    # step5: model group
    global_model, criterion = set_model(opt)
    global_model.to(opt.device)
    w_global = model_trainer.get_model_params(global_model)

    local_models = []
    for _ in range(CLIENT_NUM):
        m,_ = set_model(opt)
        model_trainer.set_model_params(m,w_global)
        local_models.append(m)
    # step6: use fedavgAPI for training
    fedavgAPI = FedAvgAPI_personal(dataset,train_label_data_local_dict, opt.device, opt, model_trainer, global_model, local_models)
    fedavgAPI.train()
    
if __name__ == '__main__':
    main()
