#coding=utf-8
from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torchvision
from torch.optim.optimizer import Optimizer, required

class Crop(namedtuple('Crop', ('h', 'w'))):
	def __call__(self, x, x0, y0):
		return x[:, y0:y0 +self.h, x0:x0 +self.w]
	
	def options(self, x_shape):
		C, H, W = x_shape
		return {'x0': range( W + 1 -self.w), 'y0': range( H + 1 -self.h)}
	
	def output_shape(self, x_shape):
		C, H, W = x_shape
		return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
	def __call__(self, x, choice):
		return x[:, :, ::-1].copy() if choice else x
	
	def options(self, x_shape):
		return {'choice': [True, False]}
	

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }