import copy
import numpy as np

from .strategy import Strategy


class LeastConfidence(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            probs = self.predict_prob(unlabel_idxs, self.net)
        elif self.args.query_model_mode == "local_only":
            local_net = self.training_local_only(label_idxs)
            probs = self.predict_prob(unlabel_idxs, local_net)
        
        U = probs.max(1)[0]
        return unlabel_idxs[U.sort()[1][:n_query]]