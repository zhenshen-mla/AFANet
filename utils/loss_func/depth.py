import os
import torch
import torch.nn as nn


class loss_huber(nn.Module):
    def __init__(self):
        super(loss_huber, self).__init__()

    def forward(self, pred, truth):
        c = pred.shape[1] 
        h = pred.shape[2] 
        w = pred.shape[3] 
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        
        t = 0.2 * torch.max(torch.abs(pred - truth))
        l1 = torch.mean(torch.mean(torch.abs(pred - truth), 1), 0)
        l2 = torch.mean(torch.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2*0.1
        else:
            return l1*0.1


class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
        
    def forward(self, pred, truth):
        c = pred.shape[1]
        h = pred.shape[2]
        w = pred.shape[3]
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        return torch.mean(torch.mean((pred - truth), 1)**2, 0)


