import random

from .strategy import Strategy


class RandomSampling(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        return random.sample(unlabel_idxs, n_query)