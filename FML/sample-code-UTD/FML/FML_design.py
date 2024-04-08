from __future__ import print_function
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import remove_nan_elements

def FeatureConstructor(f1, f2, num_positive):
    fusion_weight = np.arange(1, num_positive + 1) / 10#(0.1, 0,2, ..., 0.9)

    fused_feature = []

    for fuse_id in range(num_positive):
        temp_fuse = fusion_weight[fuse_id] * f1 + (1 - fusion_weight[fuse_id]) * f2
        fused_feature.append(temp_fuse)
    
    fused_feature = torch.stack(fused_feature, dim = 1)
    return fused_feature


## contrastive fusion loss with SupCon format: https://arxiv.org/pdf/2004.11362.pdf
class ConFusionLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConFusionLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        contrast_feature = F.normalize(contrast_feature, dim = 1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index
        # print(mask.shape)#[1151, 1152] (btz*9)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )#dig to 0, others to 1 (negative samples)

        mask = mask * logits_mask#positive samples except itself

        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self,temperature=0.5) -> None:
        super(ContrastiveLoss,self).__init__()
        self.temperature = temperature
    

    def forward(self, features, labels):
        """
        features: 特征表示，形状为 (bs, feature_dim)
        labels: 数据标签，形状为 (bs,)
        """
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # print('similarity_matrix:\n', similarity_matrix)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.T).float()
        # print('mask:\n', mask)

        # TODO: 不移除对角线上的1
        # positive_mask = mask.fill_diagonal_(0)
        positive_mask = mask
        # print('positive_mask:\n', positive_mask)

        # 计算对比损失
        logits = similarity_matrix / self.temperature
        exp_logits = torch.exp(logits) * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print('log_prob:\n', log_prob)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        # mean_log_prob_pos = remove_nan_elements(mean_log_prob_pos)# TODO: 移除nan
        # print('mean_log_prob_pos:', mean_log_prob_pos)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

# Contrast Learning Test
if __name__ == '__main__':
    f1 = torch.rand(23,128)
    f2 = torch.rand(23,128)
    fused_f = FeatureConstructor(f1,f2,10)
    print(fused_f.shape)
    mask = torch.eye(23, dtype=torch.float32)
    print(mask)
