import logging

from typing import Optional
import torch
from torch import nn
import copy
import numpy as np
from torch.utils.data import DataLoader

from FML_design import ConFusionLoss,FeatureConstructor,ContrastiveLoss,CustomContrastiveLoss
from tqdm import tqdm
import math
from custom_classifier import CustomClassifier



class MetaTrainer(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def get_model_params(self, model):
        return copy.deepcopy(model.cpu().state_dict())

    def set_model_params(self, model, model_parameters):
        model.load_state_dict(copy.deepcopy(model_parameters))

    def train(self, model, train_data, device, args, epochs=None, client_idx=None):
        # model.cuda()
        
        model.train()
        print(next(model.parameters()).device)
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
        if epochs == None:
            epochs = args.epochs

        epoch_loss = []
        for epoch in tqdm(range(epochs)):
            batch_loss = []
            iter_num = 0
            for x, y in train_data:
                print("iter_num",iter_num)
                temp_model = self.get_model_params(model)
                model.to(device)
                x = x.to(device)
                y = y.to(device)
                x1 = x[:self.batch_size].to(device)
                y1 = y[:self.batch_size].to(device)
                print(x1.device)
                output = model(x1)
                
                optimizer.zero_grad() 
                loss = criterion(output, y1.long())
                loss.backward()
                optimizer.step()
                
                x2 = x[:self.batch_size].to(device)
                y2 = y[:self.batch_size].to(device)

                output = model(x2)
                
                optimizer.zero_grad() 
                loss = criterion(output, y2.long())
                loss.backward()
                
                # temp_model = temp_model.to(device)
                self.set_model_params(model, temp_model)
                optimizer.step()
                
                # zero the parameter gradients
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
            if client_idx != None:
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
            else:
                logging.info('Epoch: {}\tLoss: {:.6f}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss)))
                
        return sum(epoch_loss) / len(epoch_loss)

    def test(self, model, test_data, device, args):
        # model = self.model

        model.to(device)
        model.eval()
        y_pre = []
        y_test = []
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                predictions = torch.argmax(pred, dim=1)
                total += y.size(0)
                correct += (predictions == y).sum().item()
                y_pre.extend(predictions.cpu()) 
                y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)
                
        return acc,y_test,y_pre


    def test_all(self, model_list, model, test_data, device, args):
        # model = self.model
        num_len = len(model_list)
        model.to(device)
        model.eval()
        
        y_pre = []
        y_test = []
        total = 0
        correct = 0
        with torch.no_grad():
            for i in range(num_len):
                self.set_model_params(model, model_list[i])
                for x, y in test_data[i]:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    predictions = torch.argmax(pred, dim=1)
                    total += y.size(0)
                    correct += (predictions == y).sum().item()
                    y_pre.extend(predictions.cpu()) 
                    y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)

        return acc,y_test,y_pre
    
    def test_user(self, model_list, model, test_data, device, test_num):
        # model = self.model
        # num_len = len(test_data)
        model.to(device)
        model.eval()
        model_len = len(model_list)
        y_pre = []
        y_test = []
        total = 0
        correct = 0
        with torch.no_grad():
            for i in range(test_num):
                for x, y in test_data[i]:
                    x = x.to(device)
                    y = y.to(device)
                    max_crt = -1000000
                    predictions = None
                    for j in range(model_len):
                        self.set_model_params(model, model_list[j])
                        pred = (model(x))
                        prediction = torch.argmax(pred, dim=1)
                        crt = (prediction == y).sum().item()
                        if crt > max_crt:
                            max_crt = crt
                            predictions = prediction
                    total += y.size(0)
                    correct += max_crt
                    y_pre.extend(predictions.cpu()) 
                    y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)

        return acc,y_test,y_pre

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False


class MyModelTrainer(object):
    
    def get_model_params(self, model):
        return copy.deepcopy(model.cpu().state_dict())

    def set_model_params(self, model, model_parameters):
        model.load_state_dict(copy.deepcopy(model_parameters))

    def train(self, model:nn.Module,label_train_data:DataLoader, train_data:DataLoader, device:Optional[str], args, epochs=None, client_idx=None) -> float:
        """_summary_

        Args:
            model (nn.Module): _description_
            train_data (DataLoader): train data loader
            device (Optional[str]): _description_
            args (_type_): _description_
            epochs (_type_, optional): _description_. Defaults to None.
            client_idx (_type_, optional): _description_. Defaults to None.

        Returns:
            float: _description_
        """
        def has_duplicate(tensor):
            unique_elements, counts = torch.unique(tensor, return_counts=True)
            return torch.any(counts > 1)

        model.to(device)
        model.train()

        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = ConFusionLoss()
        # TODO: 修改criterion_labeled以提高acc
        # label_criterion = ContrastiveLoss()
        label_criterion = CustomContrastiveLoss()

        criterion_labeled = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # TODO: 自定义简易分类器
        classifier = CustomClassifier(encoder_hidden_size=128,class_num=27)
            
        if epochs == None:
            epochs = args.epochs

        epoch_loss = []
        epoch_loss_labeled = []
        epoch_loss_unlabeled = []
        for epoch in tqdm(range(epochs)):
            batch_loss = []
            batch_loss_labeled = []
            batch_loss_unlabeled = []

            
            # unlabel data training
            for x1, x2, y in train_data:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                if args.use_classifier:
                    LABEL_DATA_NUM = math.ceil(y.shape[0]*0.2)
                    """divide training data into labeled and unlabeled"""
                    x1_labeled = x1[:LABEL_DATA_NUM]
                    x2_labeled = x2[:LABEL_DATA_NUM]
                    y_labeled = y[:LABEL_DATA_NUM]
                    x1 = x1[LABEL_DATA_NUM:]
                    x2 = x2[LABEL_DATA_NUM:]
                    y = y[LABEL_DATA_NUM:]

                """part1: unlabeled data training"""
                feature1, feature2 = model(x1,x2)
                features = FeatureConstructor(feature1, feature2,num_positive=9)
                loss_unlabeled = criterion(features, y.long())

                if args.use_classifier:
                    """part2: labeled data training"""
                    feature1, feature2 = model(x1_labeled,x2_labeled)
                    y_predict = classifier(feature1,feature2)
                    loss_labeled = criterion_labeled(y_predict,y_labeled)

                if args.use_classifier:
                    loss = loss_labeled + loss_unlabeled
                else:
                    loss = loss_unlabeled

                # zero the parameter gradients
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_loss_unlabeled.append(loss_unlabeled.item())

                if args.use_classifier:
                    batch_loss_labeled.append(loss_labeled.item())
                else:
                    batch_loss_labeled.append(0)
                    
            
            # label data training
            if args.use_labeled:
                for label_x1, label_x2, label_y in label_train_data:
                    label_x1 = label_x1.to(device)
                    label_x2 = label_x2.to(device)
                    label_y = label_y.to(device)
                    """part1: labeled data training"""
                    if has_duplicate(label_y):
                        feature1, feature2 = model(label_x1,label_x2)
                        loss_labeled_1 = label_criterion(feature1,label_y)
                        loss_labeled_2 = label_criterion(feature2,label_y)
                        loss_labeled = loss_labeled_1 + loss_labeled_2
                        # zero the parameter gradients
                        optimizer.zero_grad()                
                        loss_labeled.backward()
                        optimizer.step()
                        batch_loss_labeled.append(loss_labeled.item())
                    else:
                        print('Contians no duplicate elements!')
                    

                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss_labeled.append(sum(batch_loss_labeled) / len(batch_loss_labeled))
            epoch_loss_unlabeled.append(sum(batch_loss_unlabeled) /  len(batch_loss_unlabeled))
            
            if client_idx != None:
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
            else:
                logging.info('Epoch: {}\tLoss: {:.6f}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss)))
                
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_labeled) / len(epoch_loss_labeled), sum(epoch_loss_unlabeled) / len(epoch_loss_unlabeled)

    def test(self, model, test_data, device, args):
        # model = self.model

        model.to(device)
        model.eval()
        y_pre = []
        y_test = []
        total = 0
        correct = 0

        with torch.no_grad():
            for x1, x2, y in test_data:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                feature1, feature2 = model(x1,x2)
                pred = FeatureConstructor(feature1, feature2,num_positive=9)

                # torch.Size([1, 9, 128]) -> torch.Size([1, 128]) -> torch.Size([1])
                predictions = torch.argmax(pred, dim=1)
                mode, count = torch.mode(predictions)
                predictions = torch.tensor([mode.item()],device=device)

                total += y.size(0)
                correct += (predictions == y).sum().item()
                y_pre.extend(predictions.cpu()) 
                y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)
        return acc,y_test,y_pre


    def test_all(self, model_list, model, test_data, device, args):
        # model = self.model
        num_len = len(model_list)
        model.to(device)
        model.eval()
        
        y_pre = []
        y_test = []
        total = 0
        correct = 0
        with torch.no_grad():
            for i in range(num_len):
                self.set_model_params(model, model_list[i])
                for x, y in test_data[i]:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    predictions = torch.argmax(pred, dim=1)
                    total += y.size(0)
                    correct += (predictions == y).sum().item()
                    y_pre.extend(predictions.cpu()) 
                    y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)

        return acc,y_test,y_pre
    
    def test_user(self, model_list, model, test_data, device, test_num):
        # model = self.model
        # num_len = len(test_data)
        model.to(device)
        model.eval()
        model_len = len(model_list)
        y_pre = []
        y_test = []
        total = 0
        correct = 0
        with torch.no_grad():
            for i in range(test_num):
                for x, y in test_data[i]:
                    x = x.to(device)
                    y = y.to(device)
                    max_crt = -1000000
                    predictions = None
                    for j in range(model_len):
                        self.set_model_params(model, model_list[j])
                        pred = (model(x))
                        prediction = torch.argmax(pred, dim=1)
                        crt = (prediction == y).sum().item()
                        if crt > max_crt:
                            max_crt = crt
                            predictions = prediction
                    total += y.size(0)
                    correct += max_crt
                    y_pre.extend(predictions.cpu()) 
                    y_test.extend(y.cpu())
            acc = 100 * correct / total
            acc = np.mean(acc)

        return acc,y_test,y_pre

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
