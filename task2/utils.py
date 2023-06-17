import torch 
import numpy as np 
from torch import nn
import os
import errno


class Dataset2(torch.utils.data.Dataset):
    
    def __init__(self, dir, y_sigma=None, y_miu=None):
        data = np.load(dir, allow_pickle=1)
        self.x, self.ground_truth = np.unpackbits(data['packed_fp'], axis=1), data["values"]
        if y_sigma is None:
            self.normalize = True
            self.y_sigma = self.ground_truth.std()
            self.y_miu = self.ground_truth.mean()
            self.y = (self.ground_truth - self.y_miu)/self.y_sigma
        else:
            self.normalize = False
            self.y_sigma = y_sigma 
            self.y_miu = y_miu
            self.y = self.ground_truth
            
    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x, y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.float()
        return self.mlp(x)
    
def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise