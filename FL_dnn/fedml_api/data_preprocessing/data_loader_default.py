'''
default dataloader
used for MNIST, CIFAR-10, CIFAR-100, ImageNet
'''
import logging
import copy
import numpy as np
from collections import Counter
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from fedml_api.data_preprocessing.data_partition import partition_data
from fedml_api.data_preprocessing.datasets_default import Default_truncated
import pandas as pd

def get_model_fea(tmp, base_fea=['userId', 'date', 'queryId', 'poiId', 'itemId','isClick', 'ModuleType', 'tm']):
    """
    去除索引列和不入模型的特征列
    """
    model_fea = tmp.columns.tolist()
    for c in base_fea:
        try:
            model_fea.remove(c)
        except:
            # print("no col named", c)
            pass
    # print(len(model_fea))
    return model_fea

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader

def create_data_loaders(X_train, y_train, batch_size):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_flmt_data_alltest(train_data, test_data, batch_size, model_name):
    class_num = 2
    train_x = train_data['userId']
    userIdindex_train = train_x.value_counts().index

    train_data_local_dict = dict()
    train_data_local_num_dict = dict()
    for idx in range(500):
        data=train_data[train_data['userId']==userIdindex_train[idx]]
        model_fea = get_model_fea(data)
        label_fea = ['isClick']
        dim = len(model_fea)
        train_data_idx = np.array(data[model_fea])
        train_label_idx  = np.array(data[label_fea])
        train_data_idx = train_data_idx.astype(np.float32)
        train_label_idx = train_label_idx.astype(np.int32)
        if model_name == 'CNN':
            train_data_idx = train_data_idx.reshape(train_data_idx.shape[0],1,-1)
        train_label_idx = train_label_idx.reshape(train_label_idx.shape[0])
        train_data_local_num_dict[idx] = len(train_data_idx)
        train_data_local_dict[idx] = create_data_loaders(torch.from_numpy(train_data_idx), torch.from_numpy(train_label_idx), batch_size)
    

    model_fea = get_model_fea(test_data)
    label_fea = ['isClick']
    test_data_idx = np.array(test_data[model_fea])
    test_label_idx  = np.array(test_data[label_fea])
    test_data_idx = test_data_idx.astype(np.float32)
    test_label_idx = test_label_idx.astype(np.int32)
    if model_name == 'CNN':
        test_data_idx = test_data_idx.reshape(test_data_idx.shape[0],1,-1)
    test_label_idx = test_label_idx.reshape(test_label_idx.shape[0])
    
    test_dataloader = create_data_loaders(torch.from_numpy(test_data_idx), torch.from_numpy(test_label_idx), batch_size)
    return train_data_local_dict, test_dataloader, train_data_local_num_dict, dim, 500
    
def load_flmt_data_pertest(train_data, test_data, batch_size, model_name):
    class_num = 2
    train_x = train_data['userId']
    userIdindex_train = train_x.value_counts().index
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    num_idx = 0
    
    model_fea = get_model_fea(test_data)
    dim = len(model_fea)
    label_fea = ['isClick']
    test_data_idx_global = np.array(test_data[model_fea])
    test_label_idx_global  = np.array(test_data[label_fea])
    test_data_idx_global = test_data_idx_global.astype(np.float32)
    test_label_idx_global = test_label_idx_global.astype(np.int32)
    if model_name == 'CNN':
        test_data_idx_global = test_data_idx_global.reshape(test_data_idx_global.shape[0],1,-1)
    test_label_idx_global = test_label_idx_global.reshape(test_label_idx_global.shape[0])
    
    test_global = create_data_loaders(torch.from_numpy(test_data_idx_global), torch.from_numpy(test_label_idx_global), batch_size)
    
    for idx in range(500):
        data=train_data[train_data['userId']==userIdindex_train[idx]]
        model_fea = get_model_fea(data)
        label_fea = ['isClick']
        train_data_idx = np.array(data[model_fea])
        train_label_idx  = np.array(data[label_fea])
        
        data=test_data[test_data['userId']==userIdindex_train[idx]]
        model_fea = get_model_fea(data)
        label_fea = ['isClick']
        test_data_idx = np.array(data[model_fea])
        test_label_idx  = np.array(data[label_fea])
        
        if len(test_data_idx) > 0 and len(train_data_idx) > 0:
            train_data_idx = train_data_idx.astype(np.float32)
            train_label_idx = train_label_idx.astype(np.int32)
            test_data_idx = test_data_idx.astype(np.float32)
            test_label_idx = test_label_idx.astype(np.int32)
            if model_name == 'CNN':
                train_data_idx = train_data_idx.reshape(train_data_idx.shape[0],1,-1)
                test_data_idx = test_data_idx.reshape(test_data_idx.shape[0],1,-1)
            train_label_idx = train_label_idx.reshape(train_label_idx.shape[0])
            test_label_idx = test_label_idx.reshape(test_label_idx.shape[0])
            train_data_local_num_dict[num_idx] = len(train_data_idx)
            train_data_local_dict[num_idx] = create_data_loaders(torch.from_numpy(train_data_idx), torch.from_numpy(train_label_idx), batch_size)
            test_data_local_dict[num_idx] = create_data_loaders(torch.from_numpy(test_data_idx), torch.from_numpy(test_label_idx), batch_size)
            num_idx += 1
            
    return train_data_local_dict, test_global, test_data_local_dict, train_data_local_num_dict, dim, num_idx

def load_flmt_data_year(train_data, test_data, batch_size, model_name):
    class_num = 2
    train_x = train_data['userId']
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    num_idx = 0
    
    model_fea = get_model_fea(test_data)
    dim = len(model_fea)
    label_fea = ['isClick']
    test_data_idx_global = np.array(test_data[model_fea])
    test_label_idx_global  = np.array(test_data[label_fea])
    test_data_idx_global = test_data_idx_global.astype(np.float32)
    test_label_idx_global = test_label_idx_global.astype(np.int32)
    if model_name == 'CNN':
        test_data_idx_global = test_data_idx_global.reshape(test_data_idx_global.shape[0],1,-1)
    test_label_idx_global = test_label_idx_global.reshape(test_label_idx_global.shape[0])
    
    # test_global = create_data_loaders(torch.from_numpy(test_data_idx_global), torch.from_numpy(test_label_idx_global), batch_size)
    
    for idx in range(6):
        num_idx += 1
        t_data = train_data.loc[train_data['ageStep_'+str(idx)] == 1]
        logging.info('train ageStep_{} shape {}'.format(idx, t_data.shape))
        
        model_fea = get_model_fea(t_data)
        label_fea = ['isClick']
        train_data_idx = np.array(t_data[model_fea])
        train_label_idx  = np.array(t_data[label_fea])
        
        train_data_idx = train_data_idx.astype(np.float32)
        train_label_idx = train_label_idx.astype(np.int32)
        
        if model_name == 'CNN':
            train_data_idx = train_data_idx.reshape(train_data_idx.shape[0],1,-1)
            
        train_label_idx = train_label_idx.reshape(train_label_idx.shape[0])
        train_data_local_num_dict[idx] = len(train_data_idx)
        logging.info('ageStep_{} train data num {}'.format(idx, train_data_local_num_dict[idx]))
        train_data_local_dict[idx] = create_data_loaders(torch.from_numpy(train_data_idx), torch.from_numpy(train_label_idx), 64)
        
        t_data = test_data.loc[test_data['ageStep_'+str(idx)] == 1]
        logging.info('test ageStep_{} shape {}'.format(idx, t_data.shape))
        
        model_fea = get_model_fea(t_data)
        label_fea = ['isClick']
        test_data_idx = np.array(t_data[model_fea])
        test_label_idx  = np.array(t_data[label_fea])
        
        test_data_idx = test_data_idx.astype(np.float32)
        test_label_idx = test_label_idx.astype(np.int32)
        
        if model_name == 'CNN':
            test_data_idx = test_data_idx.reshape(test_data_idx.shape[0],1,-1)
            
        test_label_idx = test_label_idx.reshape(test_label_idx.shape[0])
        logging.info('ageStep_{} test data num {}'.format(idx, len(test_data_idx)))
        test_data_local_dict[idx] = create_data_loaders(torch.from_numpy(test_data_idx), torch.from_numpy(test_label_idx), 64)
            
    return train_data_local_dict, test_data_local_dict, train_data_local_num_dict, dim, num_idx

def load_flmt_data_year_user(train_data, test_data, batch_size, model_name):
    class_num = 2
    test_x = test_data['userId']
    userIdindex_train = test_x.value_counts().index
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    num_idx = 0
    
    model_fea = get_model_fea(test_data)
    dim = len(model_fea)
    label_fea = ['isClick']
    test_data_idx_global = np.array(test_data[model_fea])
    test_label_idx_global  = np.array(test_data[label_fea])
    test_data_idx_global = test_data_idx_global.astype(np.float32)
    test_label_idx_global = test_label_idx_global.astype(np.int32)
    if model_name == 'CNN':
        test_data_idx_global = test_data_idx_global.reshape(test_data_idx_global.shape[0],1,-1)
    test_label_idx_global = test_label_idx_global.reshape(test_label_idx_global.shape[0])
    
    test_global = create_data_loaders(torch.from_numpy(test_data_idx_global), torch.from_numpy(test_label_idx_global), batch_size)
    
    for idx in range(6):
        num_idx += 1
        t_data = train_data.loc[train_data['ageStep_'+str(idx)] == 1]
        logging.info('train ageStep_{} shape {}'.format(idx, t_data.shape))
        
        model_fea = get_model_fea(t_data)
        label_fea = ['isClick']
        train_data_idx = np.array(t_data[model_fea])
        train_label_idx  = np.array(t_data[label_fea])
        
        train_data_idx = train_data_idx.astype(np.float32)
        train_label_idx = train_label_idx.astype(np.int32)
        
        if model_name == 'CNN':
            train_data_idx = train_data_idx.reshape(train_data_idx.shape[0],1,-1)
            
        train_label_idx = train_label_idx.reshape(train_label_idx.shape[0])
        train_data_local_num_dict[idx] = len(train_data_idx)
        logging.info('ageStep_{} train data num {}'.format(idx, train_data_local_num_dict[idx]))
        train_data_local_dict[idx] = create_data_loaders(torch.from_numpy(train_data_idx), torch.from_numpy(train_label_idx), 64)

    test_num = 0
    test_data_num = 0
    for idx in range(len(userIdindex_train)):
        data=test_data[test_data['userId']==userIdindex_train[idx]]
        model_fea = get_model_fea(data)
        label_fea = ['isClick']
        test_data_idx = np.array(data[model_fea])
        test_label_idx  = np.array(data[label_fea])
        
        if len(test_data_idx) > 0 :
            test_data_idx = test_data_idx.astype(np.float32)
            test_label_idx = test_label_idx.astype(np.int32)
            if model_name == 'CNN':
                test_data_idx = test_data_idx.reshape(test_data_idx.shape[0],1,-1)
            test_label_idx = test_label_idx.reshape(test_label_idx.shape[0])
            test_data_local_dict[test_num] = create_data_loaders(torch.from_numpy(test_data_idx), torch.from_numpy(test_label_idx), 32)
            test_data_num += len(test_data_idx)
            test_num += 1
    logging.info("test data have {} user".format(test_num))
    logging.info("test user have {} data".format(test_data_num))

    return train_data_local_dict, test_data_local_dict, train_data_local_num_dict, dim, num_idx, test_num

    
def load_partition_data(args, data_dir, partition_method, client_number, batch_size):
    
    train_dataset, test_dataset, class_num = load_data(args, data_dir)
    user_train_dataidx_map, user_test_dataidx_map = partition_data(args, client_number, train_dataset, test_dataset, class_num, partition_method)
    # local_train_data_num = sum([len(user_train_dataidx_map[r]) for r in range(client_number)])
    # local_test_data_num = sum([len(user_test_dataidx_map[r]) for r in range(client_number)])

    # if args.dataset == 'imagenet':
    #     train_dataset_num = len(train_dataset)
    #     test_dataset_num = len(test_dataset)
    #     selected_train_dataset = np.random.choice([i for i in range(train_dataset_num)], 10000, replace=False)
    #     selected_test_dataset = np.random.choice([i for i in range(test_dataset_num)], 10000, replace=False)
    #     train_data_global, test_data_global = get_dataloader(args, data_dir, batch_size, train_dataidxs=selected_train_dataset, test_dataidxs=selected_test_dataset)
    # else:
    # train_data_global, test_data_global = get_dataloader(args, data_dir, batch_size)
    train_data_global, test_data_global = get_dataloader(args, train_dataset, test_dataset, batch_size)
        
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    train_data_local_num_dict = dict()
    test_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_dataidxs = user_train_dataidx_map[client_idx]
        local_train_data_num = len(train_dataidxs)
        train_data_local_num_dict[client_idx] = local_train_data_num
        logging.info("client_idx = %d, local_train_sample_number = %d" % (client_idx, local_train_data_num))
        
        test_dataidxs = user_test_dataidx_map[client_idx]
        local_test_data_num = len(test_dataidxs)
        test_data_local_num_dict[client_idx] = local_test_data_num
        logging.info("client_idx = %d, local_test_sample_number = %d" % (client_idx, local_test_data_num))

        # training batch size = 64; algorithms batch size = 32
        # train_data_local, test_data_local = get_dataloader(args, data_dir, batch_size,
        #                                          train_dataidxs=train_dataidxs, test_dataidxs=test_dataidxs)
        train_data_local, test_data_local = get_dataloader(args, train_dataset, test_dataset, batch_size,
                                                 train_dataidxs=train_dataidxs, test_dataidxs=test_dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
    show_data_distribution(-1, class_num, train_data_global, test_data_global)
    for client_idx in range(client_number):
        show_data_distribution(client_idx, class_num, train_data_local_dict[client_idx], test_data_local_dict[client_idx])
        
    return train_data_global, test_data_global, \
           train_data_local_dict, test_data_local_dict, train_data_local_num_dict, test_data_local_num_dict, class_num
           
def load_centralized_data(args, data_dir, partition_method, client_number, batch_size):
    
    train_dataset, test_dataset, class_num = load_data(args, data_dir)
    user_train_dataidx_map, user_test_dataidx_map = partition_data(args, client_number, train_dataset, test_dataset, class_num, partition_method)
    # local_train_data_num = sum([len(user_train_dataidx_map[r]) for r in range(client_number)])
    # local_test_data_num = sum([len(user_test_dataidx_map[r]) for r in range(client_number)])
    
    # if args.dataset == 'imagenet':
    #     train_dataset_num = len(train_dataset)
    #     test_dataset_num = len(test_dataset)
    #     selected_train_dataset = np.random.choice([i for i in range(train_dataset_num)], 10000, replace=False)
    #     selected_test_dataset = np.random.choice([i for i in range(test_dataset_num)], 10000, replace=False)
    #     train_data_global, test_data_global = get_dataloader(args, data_dir, batch_size, train_dataidxs=selected_train_dataset, test_dataidxs=selected_test_dataset)
    # else:
    # train_data_global, test_data_global = get_dataloader(args, data_dir, batch_size)
    train_data_global, test_data_global = get_dataloader(args, train_dataset, test_dataset, batch_size)
        
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local test dataset
    test_data_local_num_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_list = []
    train_data_centralized_num = None
    train_data_centralized = None
    train_centralized_dataidxs = []
    
    transform_train, transform_test = _data_transforms(args)

    for client_idx in range(client_number):
        test_dataidxs = user_test_dataidx_map[client_idx]
        local_test_data_num = len(test_dataidxs)
        test_data_local_num_dict[client_idx] = local_test_data_num
        logging.info("client_idx = %d, local_train_sample_number = %d" % (client_idx, len(user_train_dataidx_map[client_idx])))
        logging.info("client_idx = %d, local_test_sample_number = %d" % (client_idx, local_test_data_num))
        
        # test_data_local_dataset = Default_truncated(args, data_dir, dataidxs=test_dataidxs, train=False, transform=transform_test, download=True)
        # test_data_local = data.DataLoader(dataset=test_data_local_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_data_local = get_dataloader_single(args, test_dataset, batch_size, test_dataidxs, False)
        test_data_local_dict[client_idx] = test_data_local
        
    for client_idx in range(client_number):
        train_centralized_dataidxs.extend(user_train_dataidx_map[client_idx])
        train_data_local_num_list.append(len(user_train_dataidx_map[client_idx]))
        
    train_data_centralized_num = len(train_centralized_dataidxs)
    logging.info("centralized_sample_number = %d" % (train_data_centralized_num))

    # train_data_centralized_dataset = Default_truncated(args, data_dir, dataidxs=train_centralized_dataidxs, train=True, transform=transform_train, download=True)
        
    # assert len(train_data_centralized_dataset) == sum(train_data_local_num_list)
    # train_data_centralized = data.DataLoader(dataset=train_data_centralized_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_data_centralized = get_dataloader_single(args, train_dataset, batch_size, train_centralized_dataidxs, True)
    assert len(train_data_centralized.dataset) == sum(train_data_local_num_list)
    logging.info("train_dl_centralized number = " + str(len(train_data_centralized)))
    
    return train_data_global, test_data_global, \
           train_data_centralized, test_data_local_dict, train_data_centralized_num, test_data_local_num_dict, class_num

def _data_transforms(args):
    
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        '''
        ref https://github.com/pytorch/examples/tree/master/mnist
        '''
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif args.dataset == 'cifar10':
        '''
        ref https://github.com/kuangliu/pytorch-cifar
        '''
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    elif args.dataset == 'cifar100':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    elif args.dataset == 'cinic10':
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std),
        ])
        
        transform_test = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        
    elif args.dataset == 'imagenet':
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        image_size = 224
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    elif args.dataset == 'GTSRB':
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        image_size = 224
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        assert False
    
    return transform_train, transform_test


def load_data(args, datadir):
    
    train_transform, test_transform = _data_transforms(args)

    train_dataset = Default_truncated(args, datadir, train=True, download=True, transform=train_transform)
    test_dataset = Default_truncated(args, datadir, train=False, download=True, transform=test_transform)
    
    if args.global_dataset_selected_ratio > 0:
        
        train_dataset_num = len(train_dataset)
        test_dataset_num = len(test_dataset)
        selected_train_dataset_idx = np.random.choice([i for i in range(train_dataset_num)], int(train_dataset_num*args.global_dataset_selected_ratio), replace=False).tolist()
        selected_test_dataset_idx = np.random.choice([i for i in range(test_dataset_num)], int(test_dataset_num*args.global_dataset_selected_ratio), replace=False).tolist()
        train_dataset.data = train_dataset.data[selected_train_dataset_idx]
        train_dataset.target = train_dataset.target[selected_train_dataset_idx]
        test_dataset.data = test_dataset.data[selected_test_dataset_idx]
        test_dataset.target = test_dataset.target[selected_test_dataset_idx]
        logging.info("***************Selected {} data from global train dataset**************".format(len(train_dataset)))
        logging.info("***************Selected {} data from global test dataset**************".format(len(test_dataset)))
    
    class_num = len(np.unique(train_dataset.target))
    
    return (train_dataset, test_dataset, class_num)


def get_dataloader(args, train_dataset, test_dataset, batch_size, train_dataidxs=None, test_dataidxs=None):
    
    transform_train, transform_test = _data_transforms(args)
    
    _train_dataset = copy.deepcopy(train_dataset)
    _test_dataset = copy.deepcopy(test_dataset)
    
    _train_dataset.dataidxs = train_dataidxs
    _train_dataset.transform = transform_train
    if train_dataidxs is not None:
        _train_dataset.data = _train_dataset.data[train_dataidxs]
        _train_dataset.target = _train_dataset.target[train_dataidxs]
        
    _test_dataset.dataidxs = test_dataidxs
    _test_dataset.transform = transform_test
    if test_dataidxs is not None:
        _test_dataset.data = _test_dataset.data[test_dataidxs]
        _test_dataset.target = _test_dataset.target[test_dataidxs]
    
    train_dataloader = data.DataLoader(dataset=_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = data.DataLoader(dataset=_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_dataloader, test_dataloader

def get_dataloader_single(args, dataset, batch_size, dataidxs, train):
    
    transform_train, transform_test = _data_transforms(args)
    _dataset = copy.deepcopy(dataset)
    
    _dataset.dataidxs = dataidxs
    if train:
        _dataset.transform = transform_train
    else:
        _dataset.transform = transform_test
    if dataidxs is not None:
        _dataset.data = _dataset.data[dataidxs]
        _dataset.target = _dataset.target[dataidxs]
        
    dataloader = data.DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader

def show_data_distribution(client_idx, class_num, train_dataloader, test_dataloader):
    
    train_dataset = train_dataloader.dataset
    test_dataset = test_dataloader.dataset
    
    train_dataset_labels = Counter(train_dataset.target)
    test_dataset_labels = Counter(test_dataset.target)
    
    logging.info("Client {} sample num per class in training dataset {}".format(client_idx, train_dataset_labels))
    logging.info("Client {} sample num per class in test dataset {}".format(client_idx, test_dataset_labels))

    train_dataset_distribution = [0. for _ in range(class_num)]
    test_dataset_distribution = [0. for _ in range(class_num)]
    
    train_dataset_distribution_v = np.array(list(train_dataset_labels.values())) / np.sum(list(train_dataset_labels.values()))
    test_dataset_distribution_v = np.array(list(test_dataset_labels.values())) / np.sum(list(test_dataset_labels.values()))

    train_dataset_distribution_k = list(train_dataset_labels.keys())
    test_dataset_distribution_k = list(test_dataset_labels.keys())

    for idx, k in enumerate(train_dataset_distribution_k):
        train_dataset_distribution[k] = train_dataset_distribution_v[idx]
    for idx, k in enumerate(test_dataset_distribution_k):
        test_dataset_distribution[k] = test_dataset_distribution_v[idx]
    
    logging.info("Client {} train dataset distribution {}".format(client_idx, list(train_dataset_distribution)))
    logging.info("Client {} test dataset distribution {}".format(client_idx, list(test_dataset_distribution)))

    return train_dataset_distribution, test_dataset_distribution

def show_data_distribution_single(client_idx, class_num, train_dataloader):
    
    train_dataset = train_dataloader.dataset
    
    train_dataset_labels = Counter(train_dataset.target)
    
    train_dataset_distribution = [0. for _ in range(class_num)]
    
    train_dataset_distribution_v = np.array(list(train_dataset_labels.values())) / np.sum(list(train_dataset_labels.values()))
    
    train_dataset_distribution_k = list(train_dataset_labels.keys())
    
    for idx, k in enumerate(train_dataset_distribution_k):
        train_dataset_distribution[k] = train_dataset_distribution_v[idx]
    
    return train_dataset_distribution, train_dataset_labels
