import argparse
import logging
import os
import random
import sys
import pandas as pd
import numpy as np
import torch
import torchvision.models as models


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.utils.load_data import load_flmt_data_pertest as load_data
from fedml_api.utils.create_model import create_model
from fedml_api.utils.add_args import add_args

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI_personal
from fedml_api.standalone.fedavg.model_trainer import MyModelTrainer


if __name__ == "__main__":
    # step1: parameter definition and data loading
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    batch_size = 1024
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)
    path = "../dataset/TrainSet_before_1213.csv"
    train_data = pd.read_csv(path)
    path = "../dataset/TestSet_before_1213.csv"
    test_data = pd.read_csv(path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    
    # step2: trainer
    model_trainer = MyModelTrainer()

    # step3: load data for each client
    model_name = 'CNN'
    """
    dataset format: {[UserId1]:data1,[UserId2]:data2,...}
    dim: feature num
    """
    dataset, dim, num_client = load_data(train_data, test_data, batch_size, model_name)
    args.client_num_in_total = num_client
    
    # step4: model group
    global_model = create_model('CNN', dim, batch_size)
    w_global = model_trainer.get_model_params(global_model)
    
    local_models = []
    for _ in range(args.client_num_in_total):
        m = create_model('CNN', dim, batch_size)
        model_trainer.set_model_params(m, w_global)
        local_models.append(m)
    
    logging.info(global_model)
    # print("device",device)
    
    # step5: use fedavgAPI for training
    fedavgAPI = FedAvgAPI_personal(dataset, device, args, model_trainer, global_model, local_models)
    fedavgAPI.train()
