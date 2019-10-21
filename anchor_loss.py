#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class AnchorLoss(nn.Module):
    r"""Anchor Loss: modulates the standard cross entropy based on the prediction difficulty.
        Loss(x, y) = - y * (1 - x + p_pos)^gamma_pos * \log(x)
                        - (1 - y) * (1 + x - p_neg)^gamma_neg * \log(1-x)
    
        The losses are summed over class and averaged across observations for each minibatch.
        
        
        Args:
            gamma(float, optional): gamma > 0; reduces the relative loss for well-classiﬁed examples, 
                                    putting more focus on hard, misclassiﬁed examples
            slack(float, optional): a margin variable to penalize the output variables which are close to
                                    true positive prediction score
            warm_up(bool, optional): if ``True``, the loss is replaced to cross entropy for the first 5 epochs,
                                     and additional epoch variable which indicates the current epoch is needed
            anchor(string, optional): specifies the anchor probability type: 
                                      ``pos``: modulate target class loss
                                      ``neg``: modulate background class loss
        Shape:
            - Input: (N, C) where C is the number of classes
            - Target: (N) where each value is the class label of each sample
            - Epoch: int, optional variable when using warm_up
            - Output: scalar
            
    """
    def __init__(self, gamma=0.5, slack=0.05, anchor='neg', warm_up=False):
        super(AnchorLoss, self).__init__()
        
        assert anchor in ['neg', 'pos'], "Anchor type should be either ``neg`` or ``pos``"
        
        self.gamma = gamma
        self.slack = slack
        self.warm_up = warm_up
        self.anchor = anchor
        
        self.sig = nn.Sigmoid()
        if warm_up:
            self.ce = nn.CrossEntropyLoss().cuda()
        
        if anchor == 'pos':
            self.gamma_pos = gamma
            self.gamma_neg = 0
        elif anchor == 'neg':
            self.gamma_pos = 0
            self.gamma_neg = gamma
        
    
    def forward(self, input, target, epoch=None):
        
        if self.warm_up and not epoch: 
            raise AssertionError ("If warm_up is set to ``True``, current epoch number is required")          
        
        if self.warm_up and epoch < 5:
            loss = self.ce(input, target)
            return loss
        
        target = target.view(-1,1)
        pt = self.sig(input)
        logpt_pos = F.logsigmoid(input)
        logpt_neg = F.logsigmoid(1-input)
        
        N = input.size(0)
        C = input.size(1)
        
        class_mask = input.data.new(N, C).fill_(0)
        class_mask.scatter_(1, target.data, 1.)
        # class_mask = Variable(class_mask)  # pytorch version < 0.4.0
        class_mask = class_mask.cuda()
        class_mask = class_mask.float()

        pt_pos = pt.gather(1,target).view(-1,1)
        pt_neg = pt * (1-class_mask)
        pt_neg = pt_neg.max(dim=1)[0].view(-1,1)
        pt_neg = (pt_neg + self.slack).clamp(max=1).detach()
        pt_pos = (pt_pos - self.slack).clamp(min=0).detach()
        
        scaling_pos = -1 * (1 - pt + pt_neg).pow(self.gamma_pos)
        loss_pos = scaling_pos * logpt_pos
        scaling_neg = -1 * (1 + pt - pt_pos).pow(self.gamma_neg)
        loss_neg = scaling_neg * logpt_neg
        
        loss = class_mask * loss_pos + (1 - class_mask) * loss_neg
        loss = loss.sum(1)

        return loss.mean()

    
    
if __name__ == "__main__":
    
    AL = AnchorLoss().cuda()
    CE = nn.CrossEntropyLoss().cuda()
    
    N = 4
    C = 5
    
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    
    inputs_al = inputs.clone().cuda()
    targets_al = targets.clone().cuda()
    
    inputs_ce = inputs.clone().cuda()
    targets_ce = targets.clone().cuda()
    
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    al_loss = AL(inputs_al, targets_al)
    ce_loss = CE(inputs_ce, targets_ce)
    
    print('ce = {}, al = {}'.format(ce_loss.item(), al_loss.item()))

