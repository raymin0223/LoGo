import copy
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .strategy import Strategy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item


class DBAL(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        beta = 10
        
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            net = self.net
            probs = self.predict_prob(unlabel_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)
            probs = self.predict_prob(unlabel_idxs, net)
            
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        
        unlabel_idxs = unlabel_idxs[U.sort()[1].numpy()[:n_query * beta]]
        U = U.sort()[0].numpy()[:n_query * beta]
        
        feats = self.get_embedding(unlabel_idxs, net=net)
        
        # Avoids ValueErrors when we try to sample more instances than we have data points
        n_clusters = min(n_query, feats.shape[0])

        # Fit kmeans to data
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(feats, sample_weight=1-U)
        
        return unlabel_idxs[np.argmin(kmeans.transform(feats), axis=0)]