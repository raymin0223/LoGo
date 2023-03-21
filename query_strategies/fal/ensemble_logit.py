import pdb
import copy
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances

import torch

from ..strategy import Strategy


class EnsLogitConf(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        g_net = self.net
        l_net = self.training_local_only(label_idxs)
        
        probs1 = self.predict_prob(unlabel_idxs, g_net)
        probs2 = self.predict_prob(unlabel_idxs, l_net)
        probs = (probs1 + probs2) / 2
        
        U = probs.max(1)[0]
        
        return unlabel_idxs[U.sort()[1][:n_query]]


class EnsLogitMargin(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        g_net = self.net
        l_net = self.training_local_only(label_idxs)
        
        probs1 = self.predict_prob(unlabel_idxs, g_net)
        probs2 = self.predict_prob(unlabel_idxs, l_net)
        probs = (probs1 + probs2) / 2

        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        
        return unlabel_idxs[U.sort()[1][:n_query]]


class EnsLogitEntropy(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        g_net = self.net
        l_net = self.training_local_only(label_idxs)
        
        probs1 = self.predict_prob(unlabel_idxs, g_net)
        probs2 = self.predict_prob(unlabel_idxs, l_net)
        probs = (probs1 + probs2) / 2

        log_probs = torch.log(probs)

        log_probs[log_probs == float("-inf")] = 0
        log_probs[log_probs == float("inf")] = 0
        
        U = (probs*log_probs).sum(1)
                
        return unlabel_idxs[U.sort()[1][:n_query]]


class EnsLogitCoreSet(Strategy):
    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        self.tor = 1e-4
        
        data_idxs = list(unlabel_idxs) + list(label_idxs)
        
        unlabel_idxs = np.array(unlabel_idxs)
        label_idxs = np.array(label_idxs)
        
        g_net = self.net
        l_net = self.training_local_only(label_idxs)

        embedding1 = self.get_embedding(data_idxs, g_net)
        embedding2 = self.get_embedding(data_idxs, l_net)
        embedding = (embedding1 + embedding2) / 2
        embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[:len(unlabel_idxs), :], embedding[len(unlabel_idxs):, :], n_query)

        return unlabel_idxs[chosen]


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
#     print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
#         print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class EnsLogitBadge(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        g_net = self.net
        l_net = self.training_local_only(label_idxs)

        gradEmbedding1 = self.get_grad_embedding(list(unlabel_idxs), net=g_net)
        gradEmbedding2 = self.get_grad_embedding(list(unlabel_idxs), net=l_net)
        gradEmbedding = (gradEmbedding1 + gradEmbedding2) / 2        
        gradEmbedding = gradEmbedding.numpy()
        
        chosen = init_centers(gradEmbedding, n_query),
        
        unlabel_idxs = np.array(unlabel_idxs)
        
        return unlabel_idxs[chosen]
