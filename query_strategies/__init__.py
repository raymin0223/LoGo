import os
import sys
import copy
import pickle
import random
import datetime
import numpy as np

import torch

from models import get_model
from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .core_set import CoreSet
from .badge_sampling  import BadgeSampling
from .adversial_deepfool import AdversarialDeepFool
from .dbal import DBAL
from .egl import EGL
from .gcnal import GCNAL
from .alfa_mix import ALFAMix
from .fal import EnsLogitEntropy, EnsLogitBadge
from .fal import EnsRankEntropy, EnsRankBadge
from .fal import FTEntropy, FTBadge
from .fal import LoGo


def random_query_samples(dict_users_train_total, dict_users_test_total, args):
    """ randomly select the labeled samples at the first round
    """
    args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
            
    with open(args.dict_users_total_path, 'wb') as handle:
        pickle.dump((dict_users_train_total, dict_users_test_total), handle)
        
    dict_users_train_label_path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))

    dict_users_train_label = {user_idx: [] for user_idx in dict_users_train_total.keys()}

    # sample n_start example on each client
    for idx in dict_users_train_total.keys():
        dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), int(args.n_data / args.num_users), replace=False)
        
    with open(dict_users_train_label_path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)    
    
    return dict_users_train_label, args
    
    
def algo_query_samples(dataset_train, dataset_query, dict_users_train_total, args):
    """ query samples from the unlabeled pool
    """
    previous_ratio = args.current_ratio - args.query_ratio
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(previous_ratio))    
    with open(path, 'rb') as f:
        dict_users_train_label = pickle.load(f) 

        print("Before Querying")
        total_data_cnt = 0
        for user_idx in range(args.num_users):
            print(user_idx, len(dict_users_train_label[user_idx]))
            total_data_cnt += len(dict_users_train_label[user_idx])

        print(total_data_cnt)
        print("-" * 20)

    # Build model
    query_net = get_model(args)
    args.raw_ckpt = copy.deepcopy(query_net.state_dict())

    query_net_state_dict = torch.load(args.query_model)
    query_net.load_state_dict(query_net_state_dict)            

    # AL baselines
    if args.al_method == "random":
        strategy = RandomSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "conf":
        strategy = LeastConfidence(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "margin":
        strategy = MarginSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "entropy":
        strategy = EntropySampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "coreset":
        strategy = CoreSet(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "badge":
        strategy = BadgeSampling(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "gcnal":
        strategy = GCNAL(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "alfa_mix":
        strategy = ALFAMix(dataset_query, dataset_train, query_net, args)
        
    # FAL baselines
    elif args.al_method == "ens_logit_entropy":
        strategy = EnsLogitEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ens_logit_badge":
        strategy = EnsLogitBadge(dataset_query, dataset_train, query_net, args) 
    elif args.al_method == "ens_rank_entropy":
        strategy = EnsRankEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ens_rank_badge":
        strategy = EnsRankBadge(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ft_entropy":
        strategy = FTEntropy(dataset_query, dataset_train, query_net, args)
    elif args.al_method == "ft_badge":
        strategy = FTBadge(dataset_query, dataset_train, query_net, args)
    
    # our LoGo algorithm
    elif args.al_method == "logo":
        strategy = LoGo(dataset_query, dataset_train, query_net, args)
        
    else:
        exit('There is no al methods')    
    
    time = datetime.timedelta()
    for user_idx in dict_users_train_total.keys():                
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        
        start = datetime.datetime.now()
        new_data = strategy.query(user_idx, label_idxs, unlabel_idxs, int(args.n_query / args.num_users))
        time += datetime.datetime.now() - start
        
        print(args.al_method, user_idx)
        print("(Before) Label examples: {}".format(len(label_idxs)))
        if len(new_data) < int(args.n_query / args.num_users):
            sys.exit("too few remaining examples to query")

        dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs))   
        print("(After) Label examples: {}".format(len(list(new_data)) + len(label_idxs))) 

    time /= len(dict_users_train_total)     
    print('Querying instances takes {}'.format(time))           

    # Save dict_users for next round
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))
    with open(path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)

    return dict_users_train_label
    