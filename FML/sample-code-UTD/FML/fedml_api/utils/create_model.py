import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size = 512, projection_size = 2):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 2, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)

class CNN(nn.Module):

    def __init__(self, dim, batch_size):
        super(CNN, self).__init__()

        self.kernel_count = 512
        self.review_count = 1
        self.kernel_size = 3
        self.review_length = 1
        self.dropout_prob = 0.5
        self.cnn_out_dim = 50
        self.batch_size = batch_size
        self.conv = nn.Conv1d(
                in_channels=dim,
                out_channels=self.kernel_count,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.review_length))
        self.drop = nn.Dropout(p=self.dropout_prob)

        self.linear = nn.Sequential(
            nn.Linear(self.kernel_count * self.review_count, self.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob))
        # self.fm = FactorizationMachine(self.cnn_out_dim, 2)
        self.fm = nn.Linear(self.cnn_out_dim, 2)
        
    def forward(self, x): 
        latent = x.permute(0, 2, 1)
        latent = self.conv(latent)  
        latent = self.relu(latent)
        latent = self.maxpool(latent)
        latent = self.drop(latent)
        latent = latent.view(latent.size(0), -1)
        latent = self.linear(latent)
        latent = self.fm(latent)
        
        return latent  




def create_model(model_name, dim, batch_size):
    if model_name=='CNN':
        return CNN(dim, batch_size)
    else:
        return MLP(dim)