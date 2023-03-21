import pdb
import copy
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances

from .strategy import Strategy


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


class BadgeSampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        if self.args.query_model_mode == "global":
            gradEmbedding = self.get_grad_embedding(list(unlabel_idxs), net=self.net)
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)
            gradEmbedding = self.get_grad_embedding(list(unlabel_idxs), net=net)
        
        gradEmbedding = gradEmbedding.numpy()
        
        chosen = init_centers(gradEmbedding, n_query),
        
        unlabel_idxs = np.array(unlabel_idxs)
        return unlabel_idxs[chosen]
