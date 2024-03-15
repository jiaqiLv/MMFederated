import logging
from collections import Counter
import numpy as np
import torch
import torch.utils.data as data

def partition_data(args, num_users, train_dataset, test_dataset, class_num, partition_method):
    logging.info("*********partition data***************")
    
    if partition_method != -1:
        if partition_method == 0:
            user_train_dataidx_map = iid_partition(args, train_dataset, num_users)
            user_test_dataidx_map = iid_partition(args, test_dataset, num_users)
        elif partition_method == 1:
            user_train_dataidx_map, user_test_dataidx_map = noniid_shards_partition(args, class_num, train_dataset, test_dataset, num_users, args.num_shards_per_user, args.num_classes_per_user)
        elif partition_method == 2:
            user_train_dataidx_map, user_test_dataidx_map = noniid_Dirichlet_partition(args, train_dataset, test_dataset, num_users, class_num)
        elif partition_method == 3:
            user_train_dataidx_map, user_test_dataidx_map = noniid_class_normal(args, train_dataset, test_dataset, num_users, class_num)
        elif partition_method == 4:
            num_shards_per_user_list = np.random.randint(args.min_shards_num, args.max_shards_num, args.client_num_in_total)
            logging.info("num of shards per user {}".format(num_shards_per_user_list))
            user_train_dataidx_map, user_test_dataidx_map = noniid_shards_partition_diff(args, class_num, train_dataset, test_dataset, num_users, num_shards_per_user_list, args.num_classes_per_user)
        else:
            assert False
        
        return user_train_dataidx_map, user_test_dataidx_map
    else:
        assert False

def iid_partition(args, dataset, num_users):
    '''
    iid data partition
    each user has the same total number of samples
    '''
    if args.datasize_per_client > 0:
        num_items = args.datasize_per_client
    else:
        num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users

def noniid_shards_partition(args, class_num, train_dataset, test_dataset, num_users, num_shards_per_user, num_classes_per_user):
    '''
    dataset is partitioned into shards
    each user is allocated with {param num_shards_per_user} shards
    each user has the same total number of samples
    '''
    # if args.dataset == 'HAR':
    #     num_imgs_train = 100
    #     num_imgs_test = 25
    # elif args.dataset == 'GTSRB':
    #     num_imgs_train = 15
    #     num_imgs_test = 4
    # else:
    #     num_imgs_train = 60
    #     num_imgs_test = 15
    num_imgs_train = args.sample_num_per_shard
    num_imgs_test = int(num_imgs_train * 3 / 7)
        
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_train = np.arange(len(train_dataset))
    
    idxs_train = np.arange(len(train_dataset))
    idxs_test = np.arange(len(test_dataset))
    
    labels_train = train_dataset.target
    labels_test = test_dataset.target
    
    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:,idxs_labels_train[1,:].argsort()]
    idxs_train = idxs_labels_train[0,:]
    labels_train = idxs_labels_train[1,:]
    
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    idxs_test = idxs_labels_test[0,:]
    labels_test = idxs_labels_test[1,:]
    
    labels_start_idx_train = [list(labels_train).index(i) for i in range(class_num)]
    labels_start_idx_test = [list(labels_test).index(i) for i in range(class_num)]
    
    class_idx_num_train = list(Counter(labels_train).values())
    class_idx_num_test = list(Counter(labels_test).values())
    
    class_shards_num_train = np.array(class_idx_num_train) // num_imgs_train
    class_shards_num_test = np.array(class_idx_num_test) // num_imgs_test
    
    shards_num_train = np.sum(class_shards_num_train)
    shards_num_test = np.sum(class_shards_num_test)
    
    available_class_shard_train = [list(range(class_shards_num_train[i])) for i in range(class_num)]
    available_class_shard_test = [list(range(class_shards_num_test[i])) for i in range(class_num)]
        
            
    a = list(range(num_shards_per_user))
    shard_idx_per_class = [a[(x)*num_shards_per_user//num_classes_per_user:(x+1)*num_shards_per_user//num_classes_per_user] for x in range(num_classes_per_user)]
    selected_shards_num_per_class = [len(shard_idx_per_class[i]) for i in range(num_classes_per_user)]
    assert len(selected_shards_num_per_class) == num_classes_per_user
    assert np.sum(selected_shards_num_per_class) == num_shards_per_user
    
    for i in range(num_users):
        selected_class = (num_classes_per_user * i) % class_num
        for s in range(num_classes_per_user):
            c = (selected_class + s) % class_num
            num = selected_shards_num_per_class[s]
            selected_shards_train = np.random.choice(available_class_shard_train[c], num, replace=False)
            available_class_shard_train[c] = list(set(available_class_shard_train[c]) - set(selected_shards_train))
            selected_shards_test = np.random.choice(available_class_shard_test[c], num, replace=False)
            available_class_shard_test[c] = list(set(available_class_shard_test[c]) - set(selected_shards_test))
            
            for shard in selected_shards_train:
                dict_users_train[i] = np.concatenate((dict_users_train[i], idxs_train[labels_start_idx_train[c] + shard*num_imgs_train:labels_start_idx_train[c] + (shard+1)*num_imgs_train]), axis=0)
            for shard in selected_shards_test:
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[labels_start_idx_test[c] + shard*num_imgs_test:labels_start_idx_test[c] + (shard+1)*num_imgs_test]), axis=0)
                
        dict_users_train[i] = dict_users_train[i].tolist()
        dict_users_test[i] = dict_users_test[i].tolist()
        
    return dict_users_train, dict_users_test

def noniid_shards_partition_diff(args, class_num, train_dataset, test_dataset, num_users, num_shards_per_user_list, num_classes_per_user):
    '''
    dataset is partitioned into shards
    each user idx is allocated with {param num_shards_per_user_list[idx]} shards
    each user has different total number of samples
    '''
    # if args.dataset == 'HAR':
    #     num_imgs_train = 100
    #     num_imgs_test = 25
    # elif args.dataset == 'GTSRB':
    #     num_imgs_train = 15
    #     num_imgs_test = 4
    # else:
    #     num_imgs_train = 60
    #     num_imgs_test = 15
    
    num_imgs_train = args.sample_num_per_shard
    num_imgs_test = int(num_imgs_train * 3 / 7)
        
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_train = np.arange(len(train_dataset))
    
    idxs_train = np.arange(len(train_dataset))
    idxs_test = np.arange(len(test_dataset))
    
    labels_train = train_dataset.target
    labels_test = test_dataset.target
    
    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:,idxs_labels_train[1,:].argsort()]
    idxs_train = idxs_labels_train[0,:]
    labels_train = idxs_labels_train[1,:]
    
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1,:].argsort()]
    idxs_test = idxs_labels_test[0,:]
    labels_test = idxs_labels_test[1,:]
    
    labels_start_idx_train = [list(labels_train).index(i) for i in range(class_num)]
    labels_start_idx_test = [list(labels_test).index(i) for i in range(class_num)]
    
    class_idx_num_train = list(Counter(labels_train).values())
    class_idx_num_test = list(Counter(labels_test).values())
    
    class_shards_num_train = np.array(class_idx_num_train) // num_imgs_train
    class_shards_num_test = np.array(class_idx_num_test) // num_imgs_test
    
    shards_num_train = np.sum(class_shards_num_train)
    shards_num_test = np.sum(class_shards_num_test)
    
    available_class_shard_train = [list(range(class_shards_num_train[i])) for i in range(class_num)]
    available_class_shard_test = [list(range(class_shards_num_test[i])) for i in range(class_num)]
    
    for i in range(num_users):

        a = list(range(num_shards_per_user_list[i]))
        shard_idx_per_class = [a[(x)*num_shards_per_user_list[i]//num_classes_per_user:(x+1)*num_shards_per_user_list[i]//num_classes_per_user] for x in range(num_classes_per_user)]
        selected_shards_num_per_class = [len(shard_idx_per_class[i]) for i in range(num_classes_per_user)]

        selected_class = (num_classes_per_user * i) % class_num
        for s in range(num_classes_per_user):
            c = (selected_class + s) % class_num
            num = selected_shards_num_per_class[s]
            selected_shards_train = np.random.choice(available_class_shard_train[c], num, replace=False)
            available_class_shard_train[c] = list(set(available_class_shard_train[c]) - set(selected_shards_train))
            selected_shards_test = np.random.choice(available_class_shard_test[c], num, replace=False)
            available_class_shard_test[c] = list(set(available_class_shard_test[c]) - set(selected_shards_test))
            
            for shard in selected_shards_train:
                dict_users_train[i] = np.concatenate((dict_users_train[i], idxs_train[labels_start_idx_train[c] + shard*num_imgs_train:labels_start_idx_train[c] + (shard+1)*num_imgs_train]), axis=0)
            for shard in selected_shards_test:
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[labels_start_idx_test[c] + shard*num_imgs_test:labels_start_idx_test[c] + (shard+1)*num_imgs_test]), axis=0)
                
        dict_users_train[i] = dict_users_train[i].tolist()
        dict_users_test[i] = dict_users_test[i].tolist()
        
    return dict_users_train, dict_users_test

def noniid_Dirichlet_partition(args, train_dataset, test_dataset, num_users, class_num, alpha=0.9):
    """
    all users have samples from all classes
    the number of samples per class is sampled from Dirichlet distribution with alpha
    the total number of samples per user is different
    """
    if args.partition_alpha > 0:
        alpha = args.partition_alpha
        
    dict_users_train = {}
    dict_users_test = {}
    
    idx_batch_train = [[] for _ in range(num_users)]
    idx_batch_test = [[] for _ in range(num_users)]
    
    for class_idx in range(class_num):
        labels_idxs_train = np.where(train_dataset.target == class_idx)[0]
        labels_idxs_test = np.where(test_dataset.target == class_idx)[0]
        
        np.random.shuffle(labels_idxs_train)
        np.random.shuffle(labels_idxs_test)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        
        sampled_probabilities_train = (np.cumsum(proportions) * len(labels_idxs_train)).astype(int)[:-1]
        sampled_probabilities_test = (np.cumsum(proportions) * len(labels_idxs_test)).astype(int)[:-1]
        
        idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(labels_idxs_train, sampled_probabilities_train))]
        idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(labels_idxs_test, sampled_probabilities_test))]
        
    for j in range(num_users):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        
        dict_users_train[j] = idx_batch_train[j]
        dict_users_test[j] = idx_batch_test[j]
        
    return dict_users_train, dict_users_test

def noniid_class_normal(args, train_dataset, test_dataset, num_users, class_num, class_num_per_user=2, mu=0, sigma=2):
    '''
    each user has {param class_num_per_user} classes
    the total number of samples per user is different
    The samples assigned to users are drawn from a log-normal distribution with the parameters 𝜇 = 0 and 𝜎 = 2
    '''
    if args.num_classes_per_user > 0:
        class_num_per_user = args.num_classes_per_user
        
    # norm_proportions = np.random.lognormal(mean=mu,sigma=sigma,size=[class_num, num_users, class_num_per_user])
    
    labels_user = [[] for _ in range(class_num)]
    # labels_user_proportions = [[] for _ in range(class_num)]
    for u in range(num_users):
        for j in range(class_num_per_user):
            l = (u + j) % class_num
            labels_user[l].append(u)
    #         labels_user_proportions[l].append(norm_proportions[l][u][j])
    
    dict_users_train = {}
    dict_users_test = {}
    
    idx_batch_train = [[] for _ in range(num_users)]
    idx_batch_test = [[] for _ in range(num_users)]
    
    for class_idx in range(class_num):
        labels_idxs_train = np.where(train_dataset.target == class_idx)[0]
        labels_idxs_test = np.where(test_dataset.target == class_idx)[0]
        
        np.random.shuffle(labels_idxs_train)
        np.random.shuffle(labels_idxs_test)
        
        # labels_user_proportions[class_idx] = np.array(labels_user_proportions[class_idx])
        # proportions = labels_user_proportions[class_idx] / np.sum(labels_user_proportions[class_idx])
        proportions = np.random.lognormal(mean=mu, sigma=sigma, size=len(labels_user[class_idx]))
        proportions = proportions / proportions.sum()
        
        sampled_probabilities_train = (np.cumsum(proportions) * len(labels_idxs_train)).astype(int)[:-1]
        sampled_probabilities_test = (np.cumsum(proportions) * len(labels_idxs_test)).astype(int)[:-1]
        
        for idx_j, idx in zip(labels_user[class_idx], np.split(labels_idxs_train, sampled_probabilities_train)):
            idx_batch_train[idx_j].extend(idx.tolist())
            
        for idx_j, idx in zip(labels_user[class_idx], np.split(labels_idxs_test, sampled_probabilities_test)):
            idx_batch_test[idx_j].extend(idx.tolist())
        
    for j in range(num_users):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        
        dict_users_train[j] = idx_batch_train[j]
        dict_users_test[j] = idx_batch_test[j]
        
    return dict_users_train, dict_users_test

def per_user_partition(args):
    '''
    the glboal dataset is naturally distributed among users
    '''
    if args.dataset == 'FEMNIST':
        pass
    
def data_split(args, dataset):
    '''
    split the dataset into training dataset and test dataset
    '''
    pass
