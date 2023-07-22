#coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
from sampler import * 
import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
        
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,transform=None, target_transform=None,download=False, use_type='test', delta_path=None, model_name=None):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.phat = 0.1
        self.gen_imbalanced_data(self.img_num_list)

        # print(type(self.data.shape))
        # save delta
        self.delta_path = delta_path
        self.model_name = model_name
        self.use_type = use_type
        if (self.use_type == 'train'):
            if not os.path.exists(os.path.join(self.delta_path, self.model_name, 'delta')):
                os.makedirs(os.path.join(self.delta_path, self.model_name, 'delta'))
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.use_type == 'train':
            if not os.path.exists(os.path.join(self.delta_path, self.model_name, 'delta', str(index)+'_tensor.pt')):
                delta = torch.zeros_like(img)
                torch.save(delta, os.path.join(self.delta_path, self.model_name, 'delta', str(index)+'_tensor.pt'))
            else:
                delta = torch.load(os.path.join(self.delta_path, self.model_name, 'delta', str(index)+'_tensor.pt'))
        
            return img, target, delta, index            
        else:
            return img, target
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        new_targets = self.get_two_class(new_targets)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_two_class(self, Y):
        Y = np.array(Y)
        # for i in range(10):
        #     print(i, len(np.where(Y ==i)[0]))
        loc_0 = np.where(Y <= (self.cls_num/2-1))[0]
        loc_1 = np.where(Y > (self.cls_num/2-1))[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        for i in range(2):
            print(i, len(np.where(Y == i)[0]))
        self.phat = len(np.where(Y == 1)[0])/len(Y)
        return Y.tolist()

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def get_two_class(self, Y):
        Y = np.array(Y)
        # for i in range(10):
        #     print(i, len(np.where(Y ==i)[0]))
        loc_0 = np.where(Y <= (self.cls_num/2-1))[0]
        loc_1 = np.where(Y > (self.cls_num/2-1))[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        print('positive sample', len(loc_1))
        print('negative sample', len(loc_0))
        self.phat = len(np.where(Y == 1)[0]) / len(Y)
        return Y.tolist()