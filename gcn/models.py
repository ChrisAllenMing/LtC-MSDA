import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nclasses, dropout = 0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv_1 = GraphConvolution(nfeat, nfeat)
        self.classifier = GraphConvolution(nfeat, nclasses)

    def forward(self, x, adj):
        feat_1 = F.relu(self.conv_1(x, adj))
        feat_1 = F.dropout(feat_1, self.dropout, training=self.training)
        logit = self.classifier(feat_1, adj)

        return logit
