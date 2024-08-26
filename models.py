# -*- coding: utf-8 -*- 
from statistics import median_high
from torch.cuda import device
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import dropout
   
import torch.nn.functional as F
import pytorch_metric_learning.losses as metric_losses

def dot_similarity(features, prototypes):
    features_reshaped = features.unsqueeze(1) 
    similarity = (features_reshaped * prototypes).sum(dim=(2, 3)) 
    
    return similarity

def l2_distance(x, y):
    x = x.unsqueeze(1)
    distance = torch.norm(x - y, dim=(2,3))
    return distance

from pytorch_metric_learning import distances as dis
class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, feat_len=None, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.feat_len = feat_len
        self.num_classes = num_classes
        self.num_centers = num_centers
        if init == 'random':
            if self.feat_len == None:
                self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers ,self.feat_dim))
            else:
                self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers ,self.feat_dim, self.feat_len  ))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

        torch.nn.init.kaiming_uniform_(self.centers)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            if len(features.shape) > 2:
                dist = l2_distance(features,center)
            else:
                f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
                if center is None:
                    c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                    dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
                else:
                    c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                    dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
                dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center 
            if len(features.shape) == 2:
                dist = features.matmul(center.t())
            else:
                dist = dot_similarity(features,center)
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist,dim=2) 

        return dist


    
class FcClfNet_MPLDM(nn.Module):
    def __init__(self, embedding_net, config, l2norm=False):
        super(FcClfNet_MPLDM, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.feat_dim = 16
        self.config = config

        self.l2norm=l2norm

        self.clf3  = nn.Sequential(nn.Flatten(), nn.Linear(512, 16))
        self.clf1  = nn.Sequential()
        self.clf2  = nn.Sequential()
        self.criterion1 = MPLLoss(num_classes=embedding_net.n_classes, feat_dim=64,feat_len=35, metric='dot')
        self.criterion2 = MPLLoss(num_classes=embedding_net.n_classes, feat_dim=128,feat_len=4, metric='dot')
        self.criterion3 = MPLLoss(num_classes=embedding_net.n_classes, feat_dim=16, metric='dot')

    def forward(self, x):
        _,x1,x2,x3 = self.embedding_net.get_embeddings(x)
        
        output1 = self.clf1(x1)
        output2 = self.clf2(x2)
        output3 = self.clf3(x3)

        return output1, output2, output3


    def get_embedding(self, x):
        x_2nd, x_3rd, x_4th = self.embedding_net.get_all_embedding(x)
        output1 = x_2nd 
        output2 = x_3rd
        output3 = x_4th
        return output1, output2, output3


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


class MPLLoss(nn.CrossEntropyLoss):
    def __init__(self, num_classes, **options):
        super(MPLLoss, self).__init__()
        self.weight_pl = 0.1
        self.weight_avu = 1.0
        self.temp = 2.0
        
        self.metric = options['metric']
        if 'feat_len' in options:
            self.Dist = Dist(num_classes=num_classes, num_centers=1, feat_dim=options['feat_dim'],feat_len=options['feat_len'])
        else:
            self.Dist = Dist(num_classes=num_classes, num_centers=1, feat_dim=options['feat_dim'])

        self.points = self.Dist.centers
        self.radius1 = nn.Parameter(torch.Tensor(1))
        self.radius2 = nn.Parameter(torch.Tensor(1))
        self.radius1.data.fill_(0)
        self.radius2.data.fill_(0)

        self.radius3 = nn.Parameter(torch.Tensor(1))
        self.radius4 = nn.Parameter(torch.Tensor(1))
        self.radius3.data.fill_(1)
        self.radius4.data.fill_(-1)

        self.ContrastiveLoss = metric_losses.ContrastiveLoss(pos_margin=1, neg_margin=0)
        # self.radius =torch.Tensor(1)
        self.margin_loss = nn.MarginRankingLoss(margin=0.0)

        self.thre_1 = 0.7 
        self.thre_2 = 0.7
        self.temperature = 1

    def init_radius(self):
        self.radius1.data.fill_(0)
        self.radius2.data.fill_(0)

    def contrastiveloss(self, pos_dist, neg_dist, inverse=False):
        if not inverse:
            loss_pos = torch.nn.functional.relu(self.radius1-pos_dist) #m-s -> s>m
            loss_neg = torch.nn.functional.relu(neg_dist-self.radius2) #s+m -> s<m
        else:
            loss_pos = torch.nn.functional.relu(pos_dist-self.radius3) #s+m -> s<m 
            loss_neg = torch.nn.functional.relu(self.radius4-neg_dist) #m-s -> s>m

        low_condition = loss_pos > 0
        loss_pos_filt = loss_pos[low_condition]
        if len(loss_pos_filt)>=1:
            loss_pos_final = loss_pos_filt.mean()
        else:
            loss_pos_final = torch.sum(pos_dist * 0)

        low_condition = loss_neg > 0
        loss_neg_filt = loss_neg[low_condition]
        if len(loss_neg_filt)>=1:
            loss_neg_final = loss_neg_filt.mean()
        else:
            loss_neg_final = torch.sum(pos_dist * 0)

        return loss_pos_final+loss_neg_final

    def avu_loss_term(self, output_probs, target, annealing_coef, eps=1e-10):
        evidence = F.softplus(output_probs)
        alpha = evidence + 1

        max_probs, predicted = F.softmax(output_probs, dim=1).max(dim=1, keepdim=True)
        uncertainty = 3 / torch.sum(alpha, dim=1, keepdim=True)

        accurate_preds = (predicted.squeeze() == target).float().unsqueeze(1)
        
  
        acc_uncertain = - max_probs * torch.log(1 - uncertainty + eps) 
        inacc_certain = - (1 - max_probs) * torch.log(uncertainty + eps)
       
        avu = annealing_coef * accurate_preds * acc_uncertain + (1 - annealing_coef) * (1 - accurate_preds) * inacc_certain
        return avu.mean()

    def forward(self, x, y, labels=None, avu_loss_weight=0.5):
        similarity_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist =  similarity_dot_p
        logits =  similarity_dot_p

        if labels is None:
            return logits, 0

        
        loss = F.cross_entropy(logits / self.temp, labels)

        if len(labels.size()) == 2:
            y = labels.argmax(dim=1)
        _dis_known = dist[torch.arange(dist.size(0)), y]
        
        mask = torch.ones(dist.size(), dtype=torch.bool)
        mask[torch.arange(dist.size(0)), y] = False

        _dis_neg = dist[mask]

        loss_cont1 = self.contrastiveloss(_dis_known,_dis_neg)
        loss_cont2 = self.contrastiveloss(_dis_known,_dis_neg,inverse=True)

        loss = loss + avu_loss_weight * loss_cont1 + avu_loss_weight * loss_cont2

        return logits, loss

    def fake_loss(self, x, y=None):
        similarity_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points, metric='l2')
        dist = -similarity_dot_p
        logit = -dist

        if y == None:
            prob = F.softmax(logit, dim=1)
            loss = (prob * torch.log(prob)).sum(1).mean().exp() #entropy maximize
        else:
            loss = F.cross_entropy(logit/self.temp, y)

        return loss
        
