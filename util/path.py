#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import os
import pickle
import numpy as np


def set_result_dir(args):        
    dataset = args.dataset if 'lt' not in args.dataset else args.dataset + '_{}'.format(args.imb_ratio)
    
    if "shard" in args.partition:
        args.result_dir = '{}/save/{}/{}_num{}_C{}_r{}_le{}_rquery{:.3f}/{}_{}/seed{}/reset_{}/qmode_{}/{}/label_ratio{:.3f}/'.format(
            args.save_dir,
            dataset,
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.query_ratio,
            args.partition,
            args.num_classes_per_user,
            args.seed, 
            args.reset,
            args.query_model_mode,
            args.al_method, 
            args.current_ratio)         
        
    elif "dir" in args.partition:
        args.result_dir = '{}/save/{}/{}/{}_num{}_C{}_r{}_le{}_rquery{:.3f}/{}_{}/seed{}/reset_{}/qmode_{}/{}/label_ratio{:.3f}/'.format(
            args.save_dir,
            args.fl_algo,
            dataset, 
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.query_ratio,
            args.partition,
            args.dd_beta, 
            args.seed, 
            args.reset,
            args.query_model_mode,
            args.al_method, 
            args.current_ratio)  
        
    if args.query_ratio == args.current_ratio:
        args.query_model = None
    else:
        # use last.pt for previous ratio
        args.query_model = args.result_dir[:-6] + '{:.3f}/'.format(args.current_ratio - args.query_ratio)
        if args.custom_name is not None:
            args.query_model += "{}/".format(args.custom_name)
        args.query_model += "last.pt"
    
    if args.custom_name is not None:
        args.result_dir += "{}/".format(args.custom_name)

    if not os.path.exists(os.path.join(args.result_dir)):
        os.makedirs(os.path.join(args.result_dir), exist_ok=True)
        
    return args


def set_dict_user_path(args):
    dataset = args.dataset if 'lt' not in args.dataset else args.dataset + '{}'.format(args.imb_ratio)
    
    if "shard" in args.partition:
        args.dict_user_path = "{}/save/dict_users_{}/{}_num{}_C{}_r{}_le{}_rquery{:.3f}/{}_{}/seed{}/".format(
            args.save_dir,
            dataset,
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.query_ratio,
            args.partition,
            args.num_classes_per_user, 
            args.seed)
        
    elif "dir" in args.partition:
        args.dict_user_path = "{}/save/{}/dict_users_{}/{}_num{}_C{}_r{}_le{}_rquery{:.3f}/{}_{}/seed{}/".format(
            args.save_dir,
            args.fl_algo,
            dataset, 
            args.model,
            args.num_users, 
            args.frac, 
            args.rounds,
            args.local_ep, 
            args.query_ratio, 
            args.partition,
            args.dd_beta, 
            args.seed)
        
    # Save dict_users for next round
    args.dict_user_path = args.dict_user_path + "reset_{}/qmode_{}/{}".format(args.reset, args.query_model_mode, args.al_method)
    if args.custom_name is not None:
        args.dict_user_path += "/{}".format(args.custom_name)
        
    if not os.path.exists(args.dict_user_path):
        os.makedirs(args.dict_user_path)
        
    return args