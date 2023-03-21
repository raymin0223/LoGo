import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    
    
class EGL(Strategy):
    def query(self, user_idx, label_idxs, unlabel_idxs, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        if self.args.query_model_mode == "global":
            net = self.net
        elif self.args.query_model_mode == "local_only":
            net = self.training_local_only(label_idxs)
        
        # gradient for bias term
        loss_func = nn.CrossEntropyLoss()
        dataloader = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        norms = torch.zeros(len(unlabel_idxs))
        for x, _, idxs in tqdm(dataloader):
            x = x.to(self.args.device)
            
            output, _ = cl_net(x)
            prob = F.softmax(output, dim=1)[0, y].item()
                
            for cls in range(self.args.num_classes):
                net.zero_grad()
                y = torch.tensor([cls]).to(self.args.device)
                
                loss = loss_func(output, y)
                loss.backward()
            
                for name, param in cl_net.named_parameters():                
                    if "linear" in name and 'weight' in name:
                        norms[idxs.item()] += (param.grad.cpu().flatten()).norm() * prob      
                        
        return unlabel_idxs[norms.sort()[1][::-1][:n_query]]