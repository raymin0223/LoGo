import numpy as np

from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def adjust_learning_rate(args, r):
    lr = args.lr
    iterations = [int(args.rounds * 3 / 4)]
    
    lr_decay_epochs = []
    for it in iterations:
        lr_decay_epochs.append(int(it))
        
    steps = np.sum(r > np.asarray(lr_decay_epochs))
    if steps > 0:
        lr = lr * (args.lr_decay ** steps)
        
    return lr