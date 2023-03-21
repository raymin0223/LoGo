#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import os
import math
import pickle
import random
import numpy as np

import torch


def shard_balance(dataset, args):
    K = args.num_classes
    y_train_dict = {i: [] for i in range(K)}
    
    for idx, d in enumerate(dataset):
        if args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            y_train_dict[d[1][0]].append(idx)
        else:
            y_train_dict[d[1]].append(idx)
    
    allocate_data_cnt_dict = {i: int(len(y_train_dict[i]) / args.num_classes_per_user)  for i in range(K)}
    
    all_label_lst = list(range(K)) * args.num_classes_per_user
    
    labels_lst = []
    
    for _ in range(args.num_users):
        temp_lst = []
    
        idx = np.random.choice(all_label_lst)
        temp_lst.append(idx)

        while len(temp_lst) != args.num_classes_per_user:
            idx = np.random.choice(all_label_lst)
            if idx not in temp_lst:
                temp_lst.append(idx)

        labels_lst.append(temp_lst)
        
        for idx in temp_lst:
            del all_label_lst[all_label_lst.index(idx)]
        
    net_dataidx_map = {i: [] for i in range(args.num_users)}
    
    for user_idx, labels in enumerate(labels_lst):
        for label in labels:
            allocate_idx = np.random.choice(y_train_dict[label], allocate_data_cnt_dict[label], replace=False)
            y_train_dict[label] = list(set(y_train_dict[label]) - set(allocate_idx))
            net_dataidx_map[user_idx] += list(allocate_idx)
                
    return dict(net_dataidx_map)


def dir_balance(dataset, args, sample=None):
    """ for the fairness of annotation cost, each client has same number of samples
    """
    C = args.num_classes
    K = args.num_users
    alpha = args.dd_beta
    
    # Generate the set of clients dataset.
    clients_data = {}
    for i in range(K):
        clients_data[i] = []

    # Divide the dataset into each class of dataset.
    total_num = len(dataset)
    total_data = {}
    data_num = np.array([0 for _ in range(C)])
    for i in range(C):
        total_data[str(i)] = []
    for idx, data in enumerate(dataset):
        if args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            total_data[str(data[1][0])].append(idx)
            data_num[int(data[1][0])] += 1
        else:
            total_data[str(data[1])].append(idx)
            data_num[int(data[1])] += 1

    clients_data_num = {}
    for client in range(K):
        clients_data_num[client] = [0] * C
    
    # Distribute the data with the Dirichilet distribution.
    if sample is None:
        diri_dis = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(C))
        sample = torch.cat([diri_dis.sample().unsqueeze(0) for _ in range(K)], 0)

        # get balanced matrix
        rsum = sample.sum(1)
        csum = sample.sum(0)
        epsilon = min(1 , K / C, C / K) / 1000

        if alpha < 10:
            r, c = 1, K / C
            while (torch.any(rsum <= r - epsilon)) or (torch.any(csum <= c - epsilon)):
                sample /= sample.sum(0)
                sample /= sample.sum(1).unsqueeze(1)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        else:
            r, c = C / K, 1
            while (torch.any(abs(rsum - r) >= epsilon)) or (torch.any(abs(csum - c) >= epsilon)):
                sample = sample / sample.sum(1).unsqueeze(1)
                sample /= sample.sum(0)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        
    x = sample * torch.tensor(data_num)
    x = torch.ceil(x).long()
    x = torch.where(x <= 1, 0, x+1) if alpha < 10 else torch.where(x <= 1, 0, x)
    # print(x)
    
    print('Dataset total num', len(dataset))
    print('Total dataset class num', data_num)

    if alpha < 10:
        remain = np.inf
        nums = math.ceil(len(dataset) / K)
        i = 0
        while remain != 0:
            i += 1
            for client_idx in clients_data.keys():
                for cls in total_data.keys():
                    tmp_set = random.sample(total_data[cls], min(len(total_data[cls]), x[client_idx, int(cls)].item()))
                    
                    if len(clients_data[client_idx]) + len(tmp_set) > nums:
                        tmp_set = tmp_set[:nums-len(clients_data[client_idx])]

                    clients_data[client_idx] += tmp_set
                    clients_data_num[client_idx][int(cls)] += len(tmp_set)

                    total_data[cls] = list(set(total_data[cls])-set(tmp_set))   

            remain = sum([len(d) for _, d in total_data.items()])
            if i == 100:
                break
                
        # to make same number of samples for each client
        index = np.where(np.array([sum(clients_data_num[k]) for k in clients_data_num.keys()]) <= nums-1)[0]
        for client_idx in index:
            n = nums - len(clients_data[client_idx])
            add = 0
            for cls in total_data.keys():
                tmp_set = total_data[cls][:n-add]
                
                clients_data[client_idx] += tmp_set
                clients_data_num[client_idx][int(cls)] += len(tmp_set)
                total_data[cls] = list(set(total_data[cls])-set(tmp_set))  
                
                add += len(tmp_set)
    else:
        cumsum = x.T.cumsum(dim=1)
        for cls, data in total_data.items():
            cum = list(cumsum[int(cls)].numpy())
            tmp = np.split(np.array(data), cum)

            for client_idx in clients_data.keys():
                clients_data[client_idx] += list(tmp[client_idx])
                clients_data_num[client_idx][int(cls)] += len(list(tmp[client_idx]))

    print('clients_data_num', clients_data_num)
    print('clients_data_num', [sum(clients_data_num[k]) for k in clients_data_num.keys()])
    with open(os.path.join(args.result_dir, 'clients_data_num.pickle'), 'wb') as f:
        pickle.dump(clients_data_num, f)

    return clients_data, sample