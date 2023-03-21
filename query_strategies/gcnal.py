import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter

from .strategy import Strategy


class GCNAL(Strategy):
    def __init__(self, dataset_query, dataset_train, net, args):
        super(GCNAL, self).__init__(dataset_query, dataset_train, net, args)

        self.method = "CoreGCN"     # UncertainGCN / CoreGCN
        self.hidden_units = 128
        self.dropout_rate = 0.3
        self.LR_GCN = 1e-3
        self.WDECAY = 5e-4
        self.lambda_loss = 1.2
        self.s_margin = 0.1
        self.subset_size = 10000

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
        unlabel_idxs = np.array(unlabel_idxs)
        label_idxs = np.array(label_idxs)

        random_unlabel_idxs = np.random.choice(unlabel_idxs, min(self.subset_size, len(unlabel_idxs)), replace=False)
        data_idxs = list(random_unlabel_idxs) + list(label_idxs)

        if self.args.query_model_mode == "global":
            net = self.net
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)

        embeds = self.get_embedding(data_idxs, net)

        u_features = embeds[:len(random_unlabel_idxs), :]
        l_features = embeds[len(random_unlabel_idxs):, :]

        features = torch.cat([u_features, l_features], dim=0)
        features = nn.functional.normalize(features.to(self.args.device))

        adj = aff_to_adj(features)

        gcn_model = GCN(nfeat=features.shape[1],
                         nhid=self.hidden_units,
                         nclass=1,
                         dropout=self.dropout_rate).to(self.args.device)

        optim_gcn = optim.Adam(gcn_model.parameters(), lr=self.LR_GCN, weight_decay=self.WDECAY)

        nlbl = np.arange(0, u_features.size(0), 1)
        lbl = np.arange(u_features.size(0), features.size(0), 1)

        print('Learning Graph Convolution Network...')
        gcn_model.train()
        for _ in tqdm(range(200)):
            optim_gcn.zero_grad()
            outputs, _, _ = gcn_model(features, adj)
            loss = BCEAdjLoss(outputs, lbl, nlbl, self.lambda_loss)
            loss.backward()
            optim_gcn.step()

        gcn_model.eval()
        with torch.no_grad():
            with torch.cuda.device(self.args.device):
                inputs = features.cuda()
                #labels = binary_labels.cuda()
            scores, _, feat = gcn_model(inputs, adj)

            if self.method == "CoreGCN":
                feat = feat.detach().cpu().numpy()
                chosen = self.furthest_first(feat[nlbl, :], feat[lbl, :], n_query)
            else:
                s_margin = self.s_margin
                scores_median = np.squeeze(torch.abs(scores[nlbl] - s_margin).detach().cpu().numpy())
                chosen = np.argsort(-(scores_median))[-n_query:]

        del gcn_model, optim_gcn, feat, features
        torch.cuda.empty_cache()

        return random_unlabel_idxs[chosen]


class GCNDataset(Dataset):
    def __init__(self, features, adj, labeled):
        self.features = features
        self.labeled = labeled
        self.adj = adj

    def __getitem__(self, index):
        return self.features[index], self.adj[index], self.labeled[index]

    def __len__(self):
        return len(self.features)


def aff_to_adj(x):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)