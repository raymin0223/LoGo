import torch

from .base import FederatedLearning


class FedAvg(FederatedLearning):
    def __init__(self, args, dict_users_train_label=None):
        super().__init__(args, dict_users_train_label)

    def train(self, net, user_idx=None, lr=0.01, momentum=0.9, weight_decay=0.00001):
        net.train()

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
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    