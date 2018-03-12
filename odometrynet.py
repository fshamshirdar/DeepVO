import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class OdometryNet(nn.Module):
    def __init__(self, embeddingnet):
        super(OdometryNet, self).__init__()
        self.embeddingnet = embeddingnet
        self.fc7 = nn.Linear(in_features=2 * 9216, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5, inplace=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=1024)
        self.relu8 = nn.ReLU(inplace=True)
        self.drop8 = nn.Dropout(p=0.5, inplace=True)
        # self.fc9 = nn.Linear(in_features=4096, out_features=6)
        self.fc9 = nn.Linear(in_features=1024, out_features=1)

#        self.init_weights(3e-3)

    def init_weights(self, init_w):
        self.fc7.weight.data = fanin_init(self.fc7.weight.data.size())
        self.fc8.weight.data.uniform_(-init_w, init_w)

    def forward(self, x1, x2):
        x1 = Variable(self.embeddingnet(x1).data).cuda()
        x2 = Variable(self.embeddingnet(x2).data).cuda()
#        print(F.pairwise_distance(x1, x2, 2))
        x = torch.cat([x1, x2], dim=1)
        x = self.fc7(x)
        x = self.relu7(x)
        if (self.training):
            x = self.drop7(x)
        x = self.fc8(x)
        x = self.relu8(x)
        if (self.training):
            x = self.drop8(x)
        x = self.fc9(x)
        return x
