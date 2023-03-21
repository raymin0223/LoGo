import copy

import torch

from .base import FederatedLearning


class SCAFFOLD(FederatedLearning):
    def __init__(self, args, dict_users_train_label=None):
        super().__init__(args, dict_users_train_label)

    def init_c_nets(self, net_glob):
        self.c_nets = {}
        for i in range(self.args.num_users):
            self.c_nets[i] = copy.deepcopy(net_glob)        
        self.c_net_glob = copy.deepcopy(net_glob)

    def train(self, net, user_idx=None, lr=0.01, momentum=0.9, weight_decay=0.00001):
        net.train()

        g_net = copy.deepcopy(net)
        c_global_para = self.c_net_glob.state_dict()
        c_local_para = self.c_nets[user_idx].state_dict()
        cnt = 0

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        epoch_loss = []  
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.data_loader:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                                
                optimizer.zero_grad()
                output, emb = net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)
                cnt += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        c_new_para = self.c_nets[user_idx].state_dict()
        self.c_delta_para = copy.deepcopy(self.c_nets[user_idx].state_dict())
        global_model_para = g_net.state_dict()
        net_para = net.state_dict()

        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * self.args.lr)
            self.c_delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_nets[user_idx].load_state_dict(c_new_para)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def on_round_start(self, net_glob):
        self.total_delta = copy.deepcopy(net_glob.state_dict())
        for key in self.total_delta:
            self.total_delta[key] = 0.0

    def on_round_end(self, idxs_users):
        for key in self.total_delta:
            self.total_delta[key] /= len(idxs_users)

        c_global_para = self.c_net_glob.state_dict()
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_para[key] += self.total_delta[key]
        self.c_net_glob.load_state_dict(c_global_para)
    
    def on_user_iter_end(self):
        for key in self.total_delta:
            self.total_delta[key] += self.c_delta_para[key]   
    