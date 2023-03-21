import pdb
import copy
import numpy as np
from enum import unique
from scipy import stats
from sklearn.metrics import pairwise_distances

import torch

from ..strategy import Strategy


class EnsRankEntropy(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        g_net = self.net
        probs = self.predict_prob(unlabel_idxs, g_net)
        log_probs = torch.log(probs)

        log_probs[log_probs == float("-inf")] = 0
        log_probs[log_probs == float("inf")] = 0

        U = (probs*log_probs).sum(1)
        g_idxs = unlabel_idxs[U.sort()[1][:n_query]]

        l_net = self.training_local_only(label_idxs)
        probs = self.predict_prob(unlabel_idxs, l_net)
        log_probs = torch.log(probs)

        log_probs[log_probs == float("-inf")] = 0
        log_probs[log_probs == float("inf")] = 0

        U = (probs*log_probs).sum(1)
        l_idxs = unlabel_idxs[U.sort()[1][:n_query]]

        # rank ensemble
        unique_idxs = {i: 0 for i in list(set(g_idxs).union(set(l_idxs)))}
        print('Length of Union between global and local-only model: ', len(unique_idxs))
        
        const = 20
        for i, (g, l) in enumerate(zip(g_idxs, l_idxs)):
            rank = 1 / (const + i)
            unique_idxs[g] += rank
            unique_idxs[l] += rank
                
        selected_idxs = np.array([k for k, _ in sorted(unique_idxs.items(), key=lambda item: item[1], reverse=True)])

        return selected_idxs[:n_query]


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


class EnsRankBadge(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        g_net = self.net
        gradEmbedding = self.get_grad_embedding(list(unlabel_idxs), net=g_net)
        gradEmbedding = gradEmbedding.numpy()
        
        chosen = init_centers(gradEmbedding, n_query),
        
        g_idxs = np.array(unlabel_idxs)[chosen]

        l_net = self.training_local_only(label_idxs)
        gradEmbedding = self.get_grad_embedding(list(unlabel_idxs), net=l_net)
        gradEmbedding = gradEmbedding.numpy()
        
        chosen = init_centers(gradEmbedding, n_query),
        
        l_idxs = np.array(unlabel_idxs)[chosen]

        # rank ensemble
        unique_idxs = {i: 0 for i in list(set(g_idxs).union(set(l_idxs)))}
        const = 20
        for i, (g, l) in enumerate(zip(g_idxs, l_idxs)):
            rank = 1 / (const + i)
            unique_idxs[g] += rank
            unique_idxs[l] += rank
                
        selected_idxs = np.array([k for k, _ in sorted(unique_idxs.items(), key=lambda item: item[1], reverse=True)])
        
        return selected_idxs[:n_query]
