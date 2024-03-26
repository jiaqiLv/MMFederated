from __future__ import print_function

import os
from os.path import join,exists
# os.environ['NCCL_DEBUG'] = 'INFO'

import sys
import argparse
import time
import math
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from FML.util import AverageMeter
from FML.util import adjust_learning_rate, warmup_learning_rate
from FML.util import set_optimizer, save_model
from FML.util import ASSIGNMENT_CLASS,CLIENT_NUM
# from cosmo_design import FeatureConstructor, ConFusionLoss
from FML.FML_design import FeatureConstructor,ConFusionLoss
import FML.data_pre as data
# from cosmo_model_guide import MyUTDModelFeature, LinearClassifierAttn
from CMC.cmc_model import MyUTDModelFeature,LinearClassifierAttn

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

DATA_CLASS = [12,23,4,6,9,10] # 训练时所用到的class


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--set_subclass',type=bool,default=False,
                        help='whether to use subclass tsne')

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
    parser.add_argument('--guide_flag', type=int, default='1',
                        help='id for recording multiple runs')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='UTD-MHAD',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=27,
                        help='num_class')
    parser.add_argument('--num_train_basic', type=int, default=1,
                        help='num_train_basic')
    parser.add_argument('--num_train_unlabel_basic', type=int, default=4,
                        help='num_train_unlabel_basic')
    parser.add_argument('--label_rate', type=int, default=5,
                        help='label_rate')
    parser.add_argument('--ckpt', type=str, default='./save/Cosmo/UTD-MHAD_models/Cosmo_UTD-MHAD_MyUTDmodel_label_',
                        help='path to pre-trained model')
    # method
    parser.add_argument('--method', type=str, default='Cosmo',
                        choices=['Cosmo'], help='choose method')
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
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--model_type',type=str,default='global',
                        help='whether to test global or client model')
    parser.add_argument('--ckpt_folder',type=str,default='/code/MMFederated/FML/sample-code-UTD/FML/model/2024-03-26-02-17-27')
    parser.add_argument('--ckpt_id',type=int,default=4)
    
    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/Cosmo/{}_models'.format(opt.dataset)
    opt.tb_path = './save/Cosmo/{}_tensorboard'.format(opt.dataset)

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
    num_of_train_unlabel = (opt.num_train_unlabel_basic * (6 - opt.label_rate/5) * np.ones(opt.num_class)).astype(int)
    print("opt.num_class", opt.num_class)
    print("num_of_train_unlabel", num_of_train_unlabel)
    #load labeled train and test data
    x_train_1, x_train_2, y_train = data.load_data(opt.num_class, num_of_train_unlabel, 3, opt.label_rate)

    train_dataset = data.Multimodal_dataset(x_train_1, x_train_2, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader


def set_model(opt):
    model = MyUTDModelFeature(input_size=1)
    # classifier = LinearClassifierAttn(num_classes=opt.num_class, guide = opt.guide_flag)
    classifier = LinearClassifierAttn(num_classes=opt.num_class)
    criterion = ConFusionLoss(temperature=opt.temp)

    ## load pretrained feature encoders
    # ckpt_path = opt.ckpt + str(opt.label_rate) + '_lr_0.01_decay_0.9_bsz_32_temp_0.07_trial_0_epoch_200/last.pth'
    ckpt_path = opt.ckpt_path

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['model']
    
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)
    model = model.to("cuda:0")
    #freeze the MLP in pretrained feature encoders
    # for name, param in model.named_parameters():
    #     if "head" in name:
    #         param.requires_grad = False
        
    return model, classifier, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_list = []
    f2_list = []
    l_list = []
    end = time.time()
    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        # compute loss
        # print('input_data1.shape:', input_data1.shape)
        feature1, feature2 = model(input_data1, input_data2)
        # print('torch.norm(feature1 - feature2):', torch.norm(feature1 - feature2))
        f1_list.append(feature1)
        f2_list.append(feature2)
        l_list.append(labels)
    try:
        f1_list = torch.cat(f1_list, dim=0) 
        f2_list = torch.cat(f2_list, dim=0) 
        l_list = torch.cat(l_list, dim=0) 
    except:
        f1_list = torch.tensor(f1_list)
        f2_list = torch.tensor(f2_list) 
        l_list = torch.tensor(l_list) 
    
    f1_list = f1_list.cpu().detach().numpy()
    f2_list = f2_list.cpu().detach().numpy()
    l_list = l_list.cpu().detach().numpy()

    x = np.concatenate((f1_list * 0.5, f2_list * 0.5), axis=1)
    # x = f1_list * 0.5 + f2_list * 0.5
    lx = [l_list[i] for i in range(l_list.shape[0])] 
    print('x.shape:',x.shape)
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(x)
    
    # 为每个类别选择一个颜色
    unique_labels = np.unique(lx)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    # 创建一个散点图，每个类别的点用不同的颜色表示
    for i, label in enumerate(unique_labels):
        if opt.set_subclass:
            if label in ASSIGNMENT_CLASS[opt.client_id][0] or label in ASSIGNMENT_CLASS[opt.client_id][1]:
                plt.scatter(X_reduced[lx == label, 0], X_reduced[lx == label, 1], c=[colors[i]], label=label)
        else:
            plt.scatter(X_reduced[lx == label, 0], X_reduced[lx == label, 1], c=[colors[i]], label=label)
    
    plt.legend()
    plt.title("t-SNE visualization of multimodal embeddings")
    plt.xlabel("t-SNE axis 1")
    plt.ylabel("t-SNE axis 2")
    plt.savefig(f'./pictures/{opt.model_type}_{opt.client_id}_{opt.set_subclass}.png')
    plt.clf()

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # # build model and criterion
    # model, classifier, criterion = set_model(opt)
    # # build optimizer
    # optimizer = set_optimizer(opt, model)

    # tensorboard

    for i in range(CLIENT_NUM):
        opt.client_id = i
        if opt.model_type == 'global':
            opt.ckpt_path = join(opt.ckpt_folder,f'{opt.ckpt_id}.pth')
            model, classifier, criterion = set_model(opt)
            optimizer = set_optimizer(opt, model)
            train(train_loader, model, criterion, optimizer, 1, opt)
        elif opt.model_type == 'client':
            opt.ckpt_path = join(opt.ckpt_folder,f'client_{i}',f'{opt.ckpt_id}.pth')
            model, classifier, criterion = set_model(opt)
            optimizer = set_optimizer(opt, model)
            train(train_loader, model, criterion, optimizer, 1, opt)
    
    

if __name__ == '__main__':
    main()
