import argparse
import copy
import logging
import os
import time
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from wideresnet import WideResNet
from AUCLoss import *

from train import auc_binary

from mnist_widesesnet import *
from mnist_dataset import *
from imbalance_cifar import *


def normalize(X):
    global mu, std
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

lower_limit, upper_limit = 0.0, 1.0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--eps', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--attack_type', default='LAP2', type=str, help='LDP2, LAP2')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--out_dir', default='none', type=str)
    parser.add_argument('--method', default='ce', type=str)
    parser.add_argument('--model_dir', default='./checkpoint', type=str)
    parser.add_argument('--model_name', default='mnist_auc_new_pgd_pgd_0.1', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    return parser.parse_args()

# def get_filename(args):
#     filename = args.dataset +  '_' + args.loss_type + '_' + args.train_type + '_' + \
#                args.test_type + '_' +  \
#                str(args.lr_max)
#     print('Store Filename: ', filename)
    
#     return filename

def get_dataset_cifar10(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # args.data_dir = 'data/'
    test_dataset = IMBALANCECIFAR10(root=args.data_dir,
                                    train=False,
                                    transform=transform_test, use_type='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False)
    return test_loader, test_dataset.phat

def get_dataset_cifar100(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # args.data_dir = 'data/'
    test_dataset = IMBALANCECIFAR100(root=args.data_dir,
                                    train=False,
                                    transform=transform_test, use_type='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False)
    return test_loader, test_dataset.phat

def get_dataset_mnist(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = IMBALANCEMNIST(root=args.data_dir,
                                    train=False,
                                    transform=transform_test, use_type='test')
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False)
    return  test_loader, test_dataset.phat
       
def get_loaders(args):
    # imagenet-Lt
    if args.dataset == 'cifar10':
        test_loader, phat = get_dataset_cifar10(args)
    elif args.dataset == 'cifar100':
        test_loader, phat = get_dataset_cifar100(args)
    elif args.dataset == 'mnist':
        test_loader, phat = get_dataset_mnist(args)
    else:
        raise NotImplementedError
    return test_loader, phat

lower_limit, upper_limit = 0.0, 1.0
# Only consider L_inf attack
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, 
                device=None, criterion=None, p_hat=0.1, args=None):
    """gen adv example...
    input: model, X, y
    output: delta
    """
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
            
    # initialize perturbation
    delta = torch.zeros_like(X).to(device)
    delta.uniform_(-epsilon, epsilon)
    # pgd
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
        
        
    for _ in range(attack_iters):
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            output = model(X + delta).view_as(y)
        elif args.dataset == 'mnist':
            output = model(X + delta)
            output = output.view_as(y)
                
        index = slice(None, None, None)
            
        loss = criterion(output, y)
        loss.backward()
        grad = delta.grad.detach()
            
        d = delta[index, :, :, :]
        g = grad[index, :, :, :]
        x = X[index, :, :, :]
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[index, :, :, :] = d
        delta.grad.zero_()
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            all_loss = criterion(model(X + delta).view_as(y), y)
        elif args.dataset == 'mnist':
            tmp = model(X + delta)
            tmp = tmp.view_as(y)
            all_loss = criterion(tmp, y)
            
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    
    return max_delta

def get_min_max_delta(model, X, y, epsilon, alpha, attack_iters,
               norm="l_inf",
               device=None, 
               criterion=None,
               args=None):
    """gen adv example...
    input: model, X, y
    output: delta
    """
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    
    delta = torch.zeros_like(X).to(device)
    
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
        
    for _ in range(attack_iters):
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            output = model(normalize(X + delta)).view_as(y)
        elif args.dataset == 'mnist':
            output = model(X + delta)
            output = output.view_as(y)
            
        index = slice(None, None, None)
        
        loss = criterion(output, y)
        loss.backward()
        grad = delta.grad.detach()
        
        d = delta[index, :, :, :]
        g = grad[index, :, :, :]
        x = X[index, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[index, :, :, :] = d
        delta.grad.zero_()
        
        # 更新max loss
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            all_loss = criterion(model(normalize(X + delta)).view_as(y), y)
        elif args.dataset == 'mnist':
            tmp = model(X + delta)
            tmp = tmp.view_as(y)
            all_loss = criterion(tmp, y)
        
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    
    return max_delta

def test(test_loader, model, device, eps, step, attack_iters, args, best_state_dict=None, phat=None):
    model.eval()
    epsilon = eps/255.
    pgd_alpha = step/255.
    test_n = 0
    target, rob_output = [], []
    
    if args.attack_type == 'LAP2':
        print("====> Using LAP2 attacks")
    elif args.attack_type == 'LDP2':
        print("====> Using LDP2 attacks")
    else:
        raise NotImplementedError
    for batch_id, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.float().to(device)
        
        if args.attack_type == 'LAP2':
            # print("====> Using LAP2 attacks")
            criterion = selfloss(device)
            delta = get_min_max_delta(model=model, X=X, y=y, epsilon=epsilon, alpha=pgd_alpha, device=device, criterion=criterion,              attack_iters=attack_iters, args=args)
        elif args.attack_type == 'LDP2':
            # print("====> Using LDP2 attacks")
            if 'param_a' in best_state_dict.keys():
                param_a = best_state_dict['param_a'].cpu().detach().item()
                param_b = best_state_dict['param_b'].cpu().detach().item()
                param_alpha = best_state_dict['param_alpha'].cpu().detach().item()
                criterion = AUCLoss(imratio=phat, device=device, a=param_a, b=param_b, alpha=param_alpha)
            else:
                criterion = nn.BCELoss()
            delta = attack_pgd(model=model, X=X, y=y, epsilon=epsilon, alpha=pgd_alpha, device=device, criterion=criterion,              attack_iters=attack_iters, args=args)

        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            output = model(normalize(X + delta)).view_as(y)
        elif args.dataset == 'mnist':
            output = model(X + delta)
            output = output.view_as(y)

        test_n += y.size(0)
        target.append(y.cpu().detach().numpy())
        rob_output.append(output.cpu().detach().numpy())
    
    target = np.concatenate(target)
    rob_output = np.concatenate(rob_output)
    test_auc = auc_binary(target, rob_output)

    return test_auc

def main(args, device, dir_i):
    # get Dataset
    print('==> Get Dataset..')
    test_loader, phat = get_loaders(args)
    
    print('==> Get Model..')
    best_state_dict = torch.load(os.path.join(args.model_dir, args.model_name, 'model_best.pth'))
    # print(best_state_dict.keys())

    # Evaluation
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        model_test = WideResNet(28, 1).to(device)
        if 'state_dict' in best_state_dict.keys():
            model_test.load_state_dict(best_state_dict['state_dict'])
        else:
            model_test.load_state_dict(best_state_dict)
    elif args.dataset == 'mnist':
        model_test = WideResNet_mnist(28, 1).to(device)
        if 'state_dict' in best_state_dict.keys():
            model_test.load_state_dict(best_state_dict['state_dict'])
        else:
            model_test.load_state_dict(best_state_dict)
        
    param_a, param_b, param_alpha = 0, 0, 0
    # if 'param_a' in best_state_dict.keys():
    #     param_a = best_state_dict['param_a'].cpu().detach().item()
    #     param_b = best_state_dict['param_b'].cpu().detach().item()
    #     param_alpha = best_state_dict['param_alpha'].cpu().detach().item()
    # else:
    #     args.loss_type = 'ce'
    model_test.cuda()
    model_test.float()
    model_test.eval()

    
    
    rob_auc = test(test_loader=test_loader, model=model_test, device=device, attack_iters=args.attack_iters, args=args, eps=args.eps, step=args.alpha, best_state_dict=best_state_dict, phat=phat)
    print('MINMAX-10  auc: ', rob_auc)

if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # random
    print('==> Random..')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # init
    print('==> Init..')
    
    main(args, device, None)
