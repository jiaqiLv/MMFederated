import logging

import torch
from torch import nn
import copy
import numpy as np



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
        for epoch in range(epochs):
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

    def train(self, model, train_data, device, args, epochs=None, client_idx=None):

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
        if epochs == None:
            epochs = args.epochs

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for x, y in train_data:
                x = x.to(device)
                y = y.to(device)
                # y = y.reshape(y.shape[0])
                logits = model(x)
                loss = criterion(logits, y.long())

                # zero the parameter gradients
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
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
