#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import os
import sys
import json
import random
import copy
import pickle
import numpy as np
import pandas as pd
import medmnist
from medmnist import INFO

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import get_model
from fl_methods import get_fl_method_class
from query_strategies import random_query_samples, algo_query_samples
from util.args import args_parser
from util.path import set_result_dir, set_dict_user_path
from util.data_simulator import shard_balance, dir_balance
from util.longtail_dataset import IMBALANCECIFAR10, IMBALANCECIFAR100
from util.misc import adjust_learning_rate


def get_dataset(args):
    MEAN = {'mnist': (0.1307,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.4376821, 0.4437697, 0.47280442], 
            'cifar10': [0.485, 0.456, 0.406], 'cifar100': [0.507, 0.487, 0.441], 'pathmnist': (0.5,), 
            'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,)}
    STD = {'mnist': (0.3081,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.19803012, 0.20101562, 0.19703614], 
           'cifar10': [0.229, 0.224, 0.225], 'cifar100': [0.267, 0.256, 0.276], 'pathmnist': (0.5,),
           'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,)}
    
    if 'lt' not in args.dataset:
        noaug = [transforms.ToTensor(),
                 transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])]
        
        weakaug = [transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])]
        
        trans_noaug = transforms.Compose(noaug)
        trans_weakaug = transforms.Compose(weakaug)
        
    # standard benchmarks
    print('Load Dataset {}'.format(args.dataset))
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=trans_noaug)
    
    elif args.dataset == "fmnist":
        dataset_train = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_weakaug)
        dataset_query = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_noaug)
        dataset_test = datasets.FashionMNIST(args.data_dir, download=True, train=False, transform=trans_noaug)

    elif args.dataset == 'emnist':
        dataset_train = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.EMNIST(args.data_dir, split='byclass', train=False, download=True, transform=trans_noaug)

    elif args.dataset == 'svhn':
        dataset_train = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_weakaug)
        dataset_query = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_noaug)
        dataset_test = datasets.SVHN(args.data_dir, 'test', download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10_lt':
        dataset_train = IMBALANCECIFAR10('train', args.imb_ratio, args.data_dir)
        dataset_query = IMBALANCECIFAR10('train', args.imb_ratio, args.data_dir, train_aug=False)
        dataset_test = IMBALANCECIFAR10('test', args.imb_ratio, args.data_dir)
        
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10_lt':
        dataset_train = IMBALANCECIFAR100('train', args.imb_ratio, args.data_dir)
        dataset_query = IMBALANCECIFAR100('train', args.imb_ratio, args.data_dir, train_aug=False)
        dataset_test = IMBALANCECIFAR100('test', args.imb_ratio, args.data_dir)

    # medical benchmarks
    elif args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        DataClass = getattr(medmnist, INFO[args.dataset]['python_class'])
        
        dataset_train = DataClass(download=True, split='train', transform=trans_weakaug)
        dataset_query = DataClass(download=True, split='train', transform=trans_noaug)
        dataset_test = DataClass(download=True, split='test', transform=trans_noaug)
        
    else:
        exit('Error: unrecognized dataset')
        
    args.dataset_train = dataset_train
    args.total_data = len(dataset_train)

    if args.partition == "shard_balance":
        dict_users_train_total = shard_balance(dataset_train, args)
        dict_users_test_total = shard_balance(dataset_test, args)
    elif args.partition == "dir_balance":
        dict_users_train_total, sample = dir_balance(dataset_train, args)
        dict_users_test_total, _ = dir_balance(dataset_test, args, sample)
    
    args.n_query = round(args.total_data, -2) * args.query_ratio
    args.n_data = round(args.total_data, -2) * args.current_ratio
    
    return dataset_train, dataset_query, dataset_test, dict_users_train_total, dict_users_test_total, args


def train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args):
    results_save_path = os.path.join(args.result_dir, 'results.csv')

    fl_method = get_fl_method_class(args.fl_algo)(args, dict_users_train_label)
    if args.fl_algo == 'scaffold':
        fl_method.init_c_nets(net_glob)

    results = []   
    for round in range(args.rounds):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        lr = adjust_learning_rate(args, round)
        print("Round {}, lr: {:.6f}, momentum:{}, weight decay:{}, idx_users: {}".format(round+1, lr, args.momentum, args.weight_decay, idxs_users))

        total_data_num = sum([len(dict_users_train_label[idx]) for idx in idxs_users])
        
        fl_method.on_round_start(net_glob=net_glob)
        
        for idx in idxs_users:
            fl_method.on_user_iter_start(dataset_train, idx)
            
            net_local = copy.deepcopy(net_glob)
            w_local, loss = fl_method.train(net=net_local.to(args.device), 
                                            user_idx=idx,
                                            lr=lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)            
            loss_locals.append(copy.deepcopy(loss))
            
            fl_method.on_user_iter_end()

            w_glob = fl_method.aggregate(w_glob=w_glob, w_local=w_local, idx_user=idx, total_data_num=total_data_num)

        fl_method.on_round_end(idxs_users)
                
        net_glob.load_state_dict(w_glob, strict=False)
        acc_test, loss_test = fl_method.test(net_glob, dataset_test)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
            round+1, loss_avg, loss_test, acc_test))
        results.append(np.array([round, loss_avg, loss_test, acc_test]))
    
    last_save_path = os.path.join(args.result_dir, 'last.pt')
    torch.save(net_glob.state_dict(), last_save_path)
    
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test'])
    final_results.to_csv(results_save_path, index=False)
            
    return net_glob.state_dict()
        

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print("device:", args.device)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
        
    args = set_result_dir(args) 
    args = set_dict_user_path(args)

    # total dataset for each client
    dataset_train, dataset_query, dataset_test, dict_users_train_total, dict_users_test_total, args = get_dataset(args)
    dict_users_train_label = None
    
    while round(args.current_ratio, 2) <= args.end_ratio:
        print('[Current data ratio] %.3f' % args.current_ratio)

        net_glob = get_model(args)
   
        if args.query_ratio == args.current_ratio:
            dict_users_train_label, args = random_query_samples(dict_users_train_total, dict_users_test_total, args)
        else:
            if dict_users_train_label is None:
                path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio - args.query_ratio))
                with open(path, 'rb') as f:
                    dict_users_train_label = pickle.load(f)
                args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
                
                last_ckpt = torch.load(args.query_model)
                            
            print("Load Total Data Idxs from {}".format(args.dict_users_total_path))
            with open(args.dict_users_total_path, 'rb') as f:
                dict_users_train_total, dict_users_test_total = pickle.load(f) 
                
            dict_users_train_label = algo_query_samples(dataset_train, dataset_query, dict_users_train_total, args)
                        
        if args.reset == 'continue' and args.query_model:
            query_net_state_dict = torch.load(args.query_model)
            net_glob.load_state_dict(query_net_state_dict)

        last_ckpt = train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args)
        
        args.current_ratio += args.query_ratio
        
        # update path
        args = set_result_dir(args) 
        args = set_dict_user_path(args)