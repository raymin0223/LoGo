import copy
import numpy as np

import torch
import torch.nn as nn

from .strategy import Strategy


class MarginSampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            net = self.net
            probs = self.predict_prob(unlabel_idxs, net)
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)
            probs = self.predict_prob(unlabel_idxs, net)
            
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        
        return unlabel_idxs[U.sort()[1].numpy()[:n_query]]


