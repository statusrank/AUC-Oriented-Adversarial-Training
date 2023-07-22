from curses import noecho
import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np
import torch.nn.functional as F

class AUCLoss(nn.Module):
    def __init__(self, device=None, imratio=None,
                 a=None,
                 b=None,
                 alpha=None):
        super(AUCLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.p = imratio
        if a is not None:
            self.a = torch.tensor(a).float().to(self.device)
            self.a.requires_grad = True
        else:
            self.a = torch.tensor(0.2).float().to(self.device)
            self.a.requires_grad = True

        if b is not None:
            self.b = torch.tensor(b).float().to(self.device)
            self.b.requires_grad = True
        else:
            self.b = torch.tensor(0.2).float().to(self.device)
            self.b.requires_grad = True
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha).float().to(self.device)
            self.alpha.requires_grad = True
        else:
            self.alpha = torch.tensor(0.2).float().to(self.device)
            self.alpha.requires_grad = True
    
    def get_loss(self, y_pred, y_true):
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * (1+self.alpha) * (torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (1 == y_true).float()))) - \
               self.p * self.alpha ** 2
        return loss
    
    def forward(self, y_pred, y_true):
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]
        
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        loss = self.get_loss(y_pred, y_true)
        
        return loss

class AUCLoss_with_lamda(AUCLoss):
    def __init__(self, device=None, imratio=None,
                 a=None,
                 b=None,
                 alpha=None,
                 lambda1=None,
                 lambda2=None) -> None:
        super(AUCLoss_with_lamda, self).__init__()
        if lambda1 is not None:
            self.lambda1 = torch.tensor(lambda1).float().to(self.device)
            self.lambda1.requires_grad = True
        else:
            self.lambda1 = torch.tensor(1.0).float().to(self.device)
            self.lambda1.requires_grad = True
        
        if lambda2 is not None:
            self.lambda2 = torch.tensor(lambda2).float().to(self.device)
            self.lambda2.requires_grad = True
        else:
            self.lambda2 = torch.tensor(1.0).float().to(self.device)
            self.lambda2.requires_grad = True
    
    def forward(self, y_pred, y_true):
        if self.p is None:
            self.p = (y_true == 1).float().sum() / y_true.shape[0]
        
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        loss = self.get_loss(y_pred, y_true)
        loss = loss - self.lambda1*(self.alpha + self.a) - self.lambda2*(self.alpha - self.b + 1)
        return loss
    
    
class mLoss(AUCLoss):
    def __init__(self, device=None, imratio=None,
                 a=None,
                 b=None,
                 alpha=None,
                 lambda1=None,
                 lambda2=None) -> None:
        super(mLoss, self).__init__()
        if lambda1 is not None:
            self.lambda1 = torch.tensor(lambda1).float().to(self.device)
            self.lambda1.requires_grad = True
        else:
            self.lambda1 = torch.tensor(1.0).float().to(self.device)
            self.lambda1.requires_grad = True
        
        if lambda2 is not None:
            self.lambda2 = torch.tensor(lambda2).float().to(self.device)
            self.lambda2.requires_grad = True
        else:
            self.lambda2 = torch.tensor(1.0).float().to(self.device)
            self.lambda2.requires_grad = True
            
    def get_loss(self, y_pred, y_true, i):
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (y_true == i).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (y_true != i).float()) + \
               2 * (1+self.alpha) * (torch.mean((self.p * y_pred * (y_true != i).float() - (1 - self.p) * y_pred * (y_true == i).float()))) - \
               self.p * self.alpha ** 2
        return loss
    
    def get_loss(self, y_pred, y_true):
        loss = (1 - self.p) * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float()) + \
               self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float()) + \
               2 * (1+self.alpha) * (torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (1 == y_true).float()))) - \
               self.p * self.alpha ** 2
        return loss
    
    def forward(self, y_pred, y_true):
        loss = - self.lambda1*(self.alpha + self.a) - self.lambda2*(self.alpha - self.b + 1)
        y_true = F.one_hot(y_true.to(torch.int64), num_classes=10)
        for i in range(10):
            if self.p is None:
                self.p = (torch.argmax(y_true, -1) == i).float().sum() / y_true.shape[0]
            
            y_pred_i = y_pred[:, i]
            y_true_i = y_true[:, i]
            
            loss = loss + self.get_loss(y_pred_i, y_true_i) 
            print(loss)
        return loss        
            
    
class selfloss(nn.Module):
    def __init__(self, device=None):
        super(selfloss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        # new_y = y_true
        # new_y[new_y==0] = -1
        loss = torch.mean(-1 * (1 == y_true).float() * y_pred + (0 == y_true).float() * y_pred)
        return loss
      