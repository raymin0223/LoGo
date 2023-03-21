import copy
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item
    
    
class Strategy:
    def __init__(self, dataset_query, dataset_train, net, args):
        self.dataset_query = dataset_query
        self.dataset_train = dataset_train
        self.net = net
        self.args = args
        self.local_net_dict = {}
        self.loss_func = nn.CrossEntropyLoss()
        
    def query(self, label_idx, unlabel_idx):
        pass

    
    def predict_prob(self, unlabel_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        if net is None:
            net = self.net
            
        net.eval()
        probs = torch.zeros([len(unlabel_idxs), self.args.num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, emb = net(x)
                probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
        return probs


    def get_embedding(self, data_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        embedding = torch.zeros([len(data_idxs), net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                out, e1 = net(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    
    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim()
        net.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
        
        
    def get_grad_embedding_maxInd(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim()
        net.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                            
            return torch.Tensor(embedding)
    
    
    def training_local_only(self, label_idxs, finetune=False):
        finetune_ep = 50
        
        local_net = deepcopy(self.net)
        if not finetune: 
            # Training Local Model from the scratch
            local_net.load_state_dict(self.args.raw_ckpt)
        # else: fine-tune from global model checkpoint
        
        # train and update
        label_train = DataLoader(DatasetSplit(self.dataset_train, label_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(local_net.parameters(), 
                                    lr=self.args.lr, 
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(finetune_ep * 3 / 4)], gamma=self.args.lr_decay)
        
        # start = datetime.now()
        for epoch in range(finetune_ep):
            local_net.train()
            for images, labels, _ in label_train:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output, emb = local_net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            correct, cnt = 0., 0.
            local_net.eval()
            with torch.no_grad():
                for images, labels, _ in label_train:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    output, _ = local_net(images)
                    
                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                    cnt += len(labels)
        
                acc = correct / cnt
                if acc >= 0.99:
                    break
        
        # time = datetime.now() - start
        # print('Local-only model fine-tuning takes {}'.format(time))

        return local_net