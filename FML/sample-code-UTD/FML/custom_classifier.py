import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomClassifier(nn.Module):
    def __init__(self, encoder_hidden_size, class_num) -> None:
        super(CustomClassifier,self).__init__()
        self.linear1 = nn.Linear(encoder_hidden_size,class_num).cuda()
        self.linear2 = nn.Linear(encoder_hidden_size,class_num).cuda()

    
    def forward(self,x1,x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        return F.softmax(x1+x2,dim=1)