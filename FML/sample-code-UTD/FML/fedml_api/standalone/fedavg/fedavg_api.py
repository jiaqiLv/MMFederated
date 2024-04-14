import copy
import logging
import os
import random

import numpy as np
import torch
from datetime import datetime

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.agg import agg_FedAvg

from util import save_model


class FedAvgAPI_meta(object):
    def __init__(self, dataset, device, args, model_trainer, global_model, local_models):
        self.device = device
        self.args = args
        [train_data_local_dict, test_data_global, test_data_local_dict, train_data_local_num_dict] = dataset
        self.test_global = test_data_global
        self.val_global = None
        self.class_num = 2
        
        self.model_global = global_model

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, test_data_global, test_data_local_dict, train_data_local_dict, model_trainer, local_models)
        
        self.client_test_acc = [[] for _ in range(len(self.client_list))]
        self.client_test_loss = [[] for _ in range(len(self.client_list))]
        self.avg_client_test_loss = []
        self.avg_client_test_acc = []
        self.join_clients = self.args.client_num_in_total * self.args.join_ratio
        self.global_test_loss = []
        self.global_test_acc = []
        self.selected_clients = []
        
    def select_clients(self):
        join_clients = int(self.join_clients)
        selected_clients = list(np.random.choice(self.client_list, join_clients, replace=False))

        return selected_clients
    
    def _setup_clients(self, train_data_local_num_dict, test_data_global, test_data_local_dict, train_data_local_dict, model_trainer, local_models):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, local_models[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params(self.model_global)
        
        for idx, client in enumerate(self.client_list):
            self.model_trainer.set_model_params(client.model, w_global)
        
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            self.selected_clients = self.select_clients()
            
            w_locals = []

            for idx, client in enumerate(self.selected_clients):
                w, l = client.train()
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights and local weights
            w_global = self._aggregate(w_locals)
            
            self.model_trainer.set_model_params(self.model_global, w_global)
            for idx, client in enumerate(self.client_list):
                self.model_trainer.set_model_params(client.model, w_global)
                
            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
                self._global_test(round_idx)

            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)
                self._global_test(round_idx)
        
        logging.info('avg_client_test_acc = {}'.format(self.avg_client_test_acc))
        logging.info('avg_client_test_loss = {}'.format(self.avg_client_test_loss))
        logging.info('global_test_acc = {}'.format(self.global_test_acc))
        logging.info('global_test_loss = {}'.format(self.global_test_loss))
        logging.info('client_test_acc = {}'.format(self.client_test_acc))
        logging.info('client_test_loss = {}'.format(self.client_test_loss))
        
        # torch.save({'round': self.args.comm_round, 
        #            'state_dict': self.model_global.state_dict(),
        #            'acc': self.global_test_acc[-1]},
        #            self.args.model_save_path + self.args.model +'_' + self.args.dataset + '_fedprox.pth')


    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def _global_test(self, round_idx):
        
        logging.info("################global_test : {}".format(round_idx))

        test_acc = self.model_trainer.test(self.model_global, self.test_global, self.device, self.args)

        # test on training dataset
        # test_acc = test_global_metrics['test_correct'] / test_global_metrics['test_total']
        # test_loss = test_global_metrics['test_loss'] / test_global_metrics['test_total']
        
        stats = {'global_test_acc': test_acc}
        # wandb.log({"Global Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Global Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        
        self.global_test_acc.append(test_acc)
        # self.global_test_loss.append(test_loss)

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        local_test_acc_list = []
        local_test_loss_list = []

        for client_idx in range(self.args.client_num_in_total):
            
            # test data
            # test_local_metrics = client.local_test(True)
            test_acc, y_test, y_pre = self.client_list[client_idx].local_test()
            from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
            from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
            precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
            recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
            f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
            accuracy = round(accuracy_score(y_test, y_pre),4)
            score = roc_auc_score(y_pre, y_test)
            score = round(score, 4)
            # test on test dataset
            # test_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            # test_loss = test_local_metrics['test_loss'] / test_local_metrics['test_total']

            self.client_test_acc[client_idx].append(test_acc)
            # self.client_test_loss[client_idx].append(test_loss)

            local_test_acc_list.append(test_acc)
            info_client = {'client_id':client_idx, 'local_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
            logging.info(info_client)
            # local_test_loss_list.append(test_loss)

        avg_local_test_acc = sum(local_test_acc_list) / len(local_test_acc_list)
        # avg_local_test_loss = sum(local_test_loss_list) / len(local_test_loss_list)

        stats = {'avg_local_test_acc': avg_local_test_acc}
        # wandb.log({"Avg Local Test/Acc": avg_local_test_acc, "round": round_idx})
        # wandb.log({"Avg Local Test/Loss": avg_local_test_loss, "round": round_idx})
        logging.info(stats)
        
        self.avg_client_test_acc.append(avg_local_test_acc)
        # self.avg_client_test_loss.append(avg_local_test_loss)



class FedAvgAPI_personal(object):
    def __init__(self, dataset,train_label_data_local_dict, device, args, model_trainer, global_model, local_models):
        """
        dataset: unlabel dataset
        train_label_data_local_dict: label dataset
        """
        self.device = device
        self.args = args
        [train_data_local_dict, test_data_global, test_data_local_dict, train_data_local_num_dict] = dataset
        self.train_label_data_local_dict = train_label_data_local_dict
        self.test_global = test_data_global
        self.val_global = None
        self.class_num = 2
        
        self.model_global = global_model

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, test_data_global, test_data_local_dict, train_data_local_dict,train_label_data_local_dict, model_trainer, local_models)
        
        self.client_test_acc = [[] for _ in range(len(self.client_list))]
        self.client_test_loss = [[] for _ in range(len(self.client_list))]
        self.avg_client_test_loss = []
        self.avg_client_test_acc = []

        self.global_test_loss = []
        self.global_test_acc = []

        # logging settings
        self.formatted_time = args.formatted_time
        if not os.path.exists(f'model/{self.formatted_time}'):
            os.mkdir(f'model/{self.formatted_time}')
        logging.basicConfig(filename=f'model/{self.formatted_time}/log.log',level=logging.INFO)
        

    def _setup_clients(self, train_data_local_num_dict, test_data_global, test_data_local_dict, train_data_local_dict,train_label_data_local_dict, model_trainer, local_models):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx,train_label_data_local_dict[client_idx], train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, local_models[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params(self.model_global) # global model weight
        
        # step1: init client model weight (using global weight)
        for idx, client in enumerate(self.client_list):
            self.model_trainer.set_model_params(client.model, w_global)
        
        # step2: training
        for round_idx in range(1,self.args.comm_round+1):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = [] # record the weight of each client

            # step2.1: train individually for each client
            for idx, client in enumerate(self.client_list):
                w, l, l_labeled, l_unlabeled = client.train()
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                print(f'round_idx:{round_idx} l:{l} l_labeled:{l_labeled} l_unlabeled:{l_unlabeled}')
            
            if self.args.use_fl:
                # step2.2: update global weights and local weights
                w_global = self._aggregate(w_locals)
            
                self.model_trainer.set_model_params(self.model_global, w_global)
                for idx, client in enumerate(self.client_list):
                    self.model_trainer.set_model_params(client.model, w_global)
                
            # step2.3: test results at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            #     self._global_test(round_idx)

            # elif round_idx % self.args.frequency_of_the_test == 0:
            #     self._local_test_on_all_clients(round_idx)
            #     self._global_test(round_idx)
            
            if round_idx%self.args.save_freq==0:
                if self.args.use_fl:
                    # step2.4: save model
                    save_model(model=self.model_global,opt=self.args,epoch=round_idx,save_file=f'model/{self.formatted_time}/{round_idx}.pth')
                else:
                    for i in range(len(self.client_list)):
                        if not os.path.exists(f'model/{self.formatted_time}/client_{i}'):
                            os.mkdir(f'model/{self.formatted_time}/client_{i}')
                        save_model(model=self.client_list[i].model,opt=self.args,epoch=round_idx,save_file=f'model/{self.formatted_time}/client_{i}/{round_idx}.pth')
        
        logging.info('avg_client_test_acc = {}'.format(self.avg_client_test_acc))
        logging.info('avg_client_test_loss = {}'.format(self.avg_client_test_loss))
        logging.info('global_test_acc = {}'.format(self.global_test_acc))
        logging.info('global_test_loss = {}'.format(self.global_test_loss))
        logging.info('client_test_acc = {}'.format(self.client_test_acc))
        logging.info('client_test_loss = {}'.format(self.client_test_loss))
        
       

    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def _global_test(self, round_idx):
        
        logging.info("################global_test : {}".format(round_idx))

        test_acc = self.model_trainer.test(self.model_global, self.test_global, self.device, self.args)
        
        stats = {'global_test_acc': test_acc}

        logging.info(stats)
        
        self.global_test_acc.append(test_acc)
        # self.global_test_loss.append(test_loss)

    def _local_test_on_all_clients(self, round_idx):
        # TODO: 修改local test的机制
        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        local_test_acc_list = []
        local_test_loss_list = []

        for client_idx in range(self.args.client_num_in_total):
            
            # test data
            # test_local_metrics = client.local_test(True)
            test_acc, y_test, y_pre = self.client_list[client_idx].local_test()
            from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
            from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer         
            
            precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
            recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
            f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
            accuracy = round(accuracy_score(y_test, y_pre),4)
            
            # TODO: 暂时不测试ROC
            score = 0.0
            # score = roc_auc_score(y_pre, y_test)
            # score = round(score, 4)
            
            # test on test dataset
            # test_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            # test_loss = test_local_metrics['test_loss'] / test_local_metrics['test_total']

            self.client_test_acc[client_idx].append(test_acc)
            # self.client_test_loss[client_idx].append(test_loss)

            local_test_acc_list.append(test_acc)
            info_client = {'client_id':client_idx, 'local_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
            print('info_client:', info_client)
            logging.info(info_client)
            # local_test_loss_list.append(test_loss)

        avg_local_test_acc = sum(local_test_acc_list) / len(local_test_acc_list)
        # avg_local_test_loss = sum(local_test_loss_list) / len(local_test_loss_list)

        stats = {'avg_local_test_acc': avg_local_test_acc}
        # wandb.log({"Avg Local Test/Acc": avg_local_test_acc, "round": round_idx})
        # wandb.log({"Avg Local Test/Loss": avg_local_test_loss, "round": round_idx})
        logging.info(stats)
        
        self.avg_client_test_acc.append(avg_local_test_acc)
        # self.avg_client_test_loss.append(avg_local_test_loss)
        
class FedAvgAPI_year(object):
    def __init__(self, dataset, device, args, model_trainer, local_models, model_global):
        self.device = device
        self.args = args
        [train_data_local_dict, test_data_local_dict, train_data_local_num_dict, test_num] = dataset
        self.val_global = None
        self.class_num = 2
        self.model_global = model_global
        self.model_list = copy.deepcopy(local_models)
        self.test_num = test_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.model_trainer = model_trainer
        self.test_data_local_dict = test_data_local_dict
        
        self._setup_clients(train_data_local_num_dict, test_data_local_dict, train_data_local_dict, model_trainer, local_models)
        
        self.client_test_acc = [[] for _ in range(len(self.client_list))]
        self.client_test_loss = [[] for _ in range(len(self.client_list))]
        self.avg_client_test_loss = []
        self.avg_client_test_acc = []

        self.global_test_loss = []
        self.global_test_acc = []
        

    def _setup_clients(self, train_data_local_num_dict, test_data_local_dict, train_data_local_dict, model_trainer, local_models):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, local_models[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):

        for round_idx in range(self.args.comm_round):

            logging.info("################ Train round : {}".format(round_idx))


            for idx, client in enumerate(self.client_list):
                w, l = client.train()
                self.model_list[idx] = copy.deepcopy(w)
            for idx, client in enumerate(self.client_list):
                self.model_trainer.set_model_params(client.model, self.model_list[idx])
            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                # self._local_test_on_all_clients(round_idx)
                # self._global_test_year(round_idx)
                self._global_test_user(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                # self._local_test_on_all_clients(round_idx)
                # self._global_test_year(round_idx)
                self._global_test_user(round_idx)
        
        logging.info('avg_client_test_acc = {}'.format(self.avg_client_test_acc))
        logging.info('avg_client_test_loss = {}'.format(self.avg_client_test_loss))
        logging.info('global_test_acc = {}'.format(self.global_test_acc))
        logging.info('global_test_loss = {}'.format(self.global_test_loss))
        logging.info('client_test_acc = {}'.format(self.client_test_acc))
        logging.info('client_test_loss = {}'.format(self.client_test_loss))
        
        # torch.save({'round': self.args.comm_round, 
        #            'state_dict': self.model_global.state_dict(),
        #            'acc': self.global_test_acc[-1]},
        #            self.args.model_save_path + self.args.model +'_' + self.args.dataset + '_fedprox.pth')


    def _aggregate(self, w_locals):
        return agg_FedAvg(w_locals)

    def _global_test_year(self, round_idx):
        
        logging.info("################global_test : {}".format(round_idx))

        test_acc, y_test, y_pre = self.model_trainer.test_all(self.model_list, self.model_global, self.test_data_local_dict, self.device, self.args)

        from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
        from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
        precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
        recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
        f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
        accuracy = round(accuracy_score(y_test, y_pre),4)
        score = roc_auc_score(y_pre, y_test)
        score = round(score, 4)

        # self.client_test_loss[client_idx].append(test_loss)

        info_client = {'global_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
        logging.info(info_client)
        
        self.global_test_acc.append(test_acc)
        # self.global_test_loss.append(test_loss)

    def _global_test_user(self, round_idx):
        
        logging.info("################global_test : {}".format(round_idx))

        test_acc, y_test, y_pre = self.model_trainer.test_user(self.model_list, self.model_global, self.test_data_local_dict, self.device, self.test_num)

        from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
        from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
        precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
        recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
        f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
        accuracy = round(accuracy_score(y_test, y_pre),4)
        score = roc_auc_score(y_pre, y_test)
        score = round(score, 4)

        # self.client_test_loss[client_idx].append(test_loss)

        info_client = {'global_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
        logging.info(info_client)
        
        self.global_test_acc.append(test_acc)
        # self.global_test_loss.append(test_loss)


    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        local_test_acc_list = []
        local_test_loss_list = []

        for client_idx in range(self.args.client_num_in_total):
            
            # test data
            # test_local_metrics = client.local_test(True)
            test_acc, y_test, y_pre = self.client_list[client_idx].local_test()
            from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, accuracy_score, f1_score
            from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
            precisionScore = round(precision_score(y_test, y_pre, average='weighted'), 4)
            recallScore = round(recall_score(y_test, y_pre, average='weighted'),4)
            f1Score = round(f1_score(y_test, y_pre, average='weighted'),4)
            accuracy = round(accuracy_score(y_test, y_pre),4)
            score = roc_auc_score(y_pre, y_test)
            score = round(score, 4)

            self.client_test_acc[client_idx].append(test_acc)
            # self.client_test_loss[client_idx].append(test_loss)

            local_test_acc_list.append(test_acc)
            info_client = {'client_id':client_idx, 'local_test_acc': test_acc, '预测精确率为':precisionScore, '预测准确率为': accuracy, '预测f1-score为': f1Score, '预测AUC为' : score, '预测召回率为': recallScore}
            logging.info(info_client)
            # local_test_loss_list.append(test_loss)

        avg_local_test_acc = sum(local_test_acc_list) / len(local_test_acc_list)
        # avg_local_test_loss = sum(local_test_loss_list) / len(local_test_loss_list)

        stats = {'avg_local_test_acc': avg_local_test_acc}
        # wandb.log({"Avg Local Test/Acc": avg_local_test_acc, "round": round_idx})
        # wandb.log({"Avg Local Test/Loss": avg_local_test_loss, "round": round_idx})
        logging.info(stats)
        
        self.avg_client_test_acc.append(avg_local_test_acc)
        # self.avg_client_test_loss.append(avg_local_test_loss)