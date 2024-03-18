import os
import random
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.data_loader_default import load_flmt_data_alltest, load_flmt_data_pertest, load_flmt_data_year, load_flmt_data_year_user

def load_data(train_data, test_data, batch_size, model_name):
    
    train_data_local_dict, test_data, train_data_local_num_dict, dim, num_client = load_flmt_data_alltest(train_data, test_data, batch_size, model_name)

    dataset = [train_data_local_dict, test_data, train_data_local_num_dict]
    return dataset, dim, num_client

def load_data_per(train_data, test_data, batch_size, model_name):
    
    train_data_local_dict, test_data, test_data_local_dict, train_data_local_num_dict, dim, num_client = load_flmt_data_pertest(train_data, test_data, batch_size, model_name)

    dataset = [train_data_local_dict, test_data, test_data_local_dict, train_data_local_num_dict]
    return dataset, dim, num_client

def load_data_year(train_data, test_data, batch_size, model_name):
    
    train_data_local_dict, test_data_local_dict, train_data_local_num_dict, dim, num_client = load_flmt_data_year(train_data, test_data, batch_size, model_name)

    dataset = [train_data_local_dict, test_data_local_dict, train_data_local_num_dict, num_client]
    return dataset, dim, num_client

def load_data_year_user(train_data, test_data, batch_size, model_name):
    
    train_data_local_dict, test_data_local_dict, train_data_local_num_dict, dim, num_client, test_num = load_flmt_data_year_user(train_data, test_data, batch_size, model_name)

    dataset = [train_data_local_dict, test_data_local_dict, train_data_local_num_dict, test_num]
    return dataset, dim, num_client