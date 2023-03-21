import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
    
    
class AdversarialDeepFool(Strategy):
    def __init__(self, dataset_query, dataset_train, net, args):
        super(AdversarialDeepFool, self).__init__(dataset_query, dataset_train, net, args)
        self.max_iter = 10

    def cal_dis(self, net, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, e1 = net(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = net(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).numpy().sum()

    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabeld_data = DatasetSplit(self.dataset_query, unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            net = self.net
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)
        
        net.cpu()
        net.eval()
        dis = np.zeros(len(unlabel_idxs))

        for i in range(len(unlabeld_data)):
            x, _, _ = unlabeld_data[i]
            dis[i] = self.cal_dis(net, x)

        net.cuda()

        unlabel_idxs = np.array(unlabel_idxs)
        return unlabel_idxs[dis.argsort()[:n_query]]