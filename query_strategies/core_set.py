import copy
import numpy as np
from sklearn.metrics import pairwise_distances

from .strategy import Strategy


class CoreSet(Strategy):
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
        data_idxs = list(unlabel_idxs) + list(label_idxs)
        
        unlabel_idxs = np.array(unlabel_idxs)
        label_idxs = np.array(label_idxs)
        
        if self.args.query_model_mode == "global":
            embedding = self.get_embedding(data_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            local_net = self.training_local_only(label_idxs)
            embedding = self.get_embedding(data_idxs, local_net)
        
        embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[:len(unlabel_idxs), :], embedding[len(unlabel_idxs):, :], n_query)

        return unlabel_idxs[chosen]