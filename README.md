# AUC-Oriented Adversarial Training

This is an official code of our AUC-Oriented Adversarial Training studies 
- AdAUC: End-to-end Adversarial AUC Optimization Against Long-tail Problems. **ICML'2022**.
- Revisiting AUC-Oriented Adversarial Training with Loss-Agnostic Perturbations. 

## Dependencies
- python 3.8+
- pytorch 1.8+
- numpy
- tqdm
- pandas
- scikit-learn
- pickle
- torchvision

## How to train
During the training phase, you should **carefully set following params** (get_parameters() function in train.py):
- dataset: support "mnist", "mnist_balance", "cifar10" and "cifar100". The other datasets will be released as soon as possible!
- lr_max: learning rate
- epoch
- epsilon, pgd-alpha, attack_iters: attack strength, step size, and step in AT 
- loss_type: support "auc" and "auc_new", where "auc" is our ICML'22 method and "auc_new" is the new algorithm

For example, in terms of imbalanced mnist dataset, you can run as follows
```python
python3 train.py --dataset=mnist --lr_max=0.1 --epoch=200 --epsilon=8.0 --loss_type=auc_new
python3 train.py --dataset=mnist --lr_max=0.01 --epoch=200 --epsilon=8.0 --loss_type=auc
```
Besides, to balanced version mnist, you can do
```python
python3 train.py --dataset=mnist_balance --lr_max=0.1 --epoch=200 --epsilon=8.0 --loss_type=auc_new
python3 train.py --dataset=mnist_balance --lr_max=0.01 --epoch=200 --epsilon=8.0 --loss_type=auc
```

## How to test 
During the test phase, you should **carefully set following params** (get_args() function in test.py):
- eps, alpha, attack_iters: attack strength, step size, and step in PGD adversarial attacks
- attack_type: the attack manner, i.e., LDP2 or LAP2 attack
- dataset
- model_name: the best model path saved in your exps.

For examples:
```python
python3 test.py --eps=8 --alpha=2 --attack_iters=10 --attack_type=LAP2 --dataset=mnist --model_name=mnist_auc_new_pgd_pgd_0.1
python3 test.py --eps=8 --alpha=2 --attack_iters=10 --attack_type=LDP2 --dataset=mnist --model_name=mnist_auc_new_pgd_pgd_0.1
```