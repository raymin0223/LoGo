import copy
import numpy as np

import torch

from .strategy import Strategy


class EntropySampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            probs = self.predict_prob(unlabel_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            local_net = self.training_local_only(label_idxs)
            probs = self.predict_prob(unlabel_idxs, local_net)
        
        log_probs = torch.log(probs)

        log_probs[log_probs == float("-inf")] = 0
        log_probs[log_probs == float("inf")] = 0

        U = (probs*log_probs).sum(1)
        return unlabel_idxs[U.sort()[1][:n_query]]
