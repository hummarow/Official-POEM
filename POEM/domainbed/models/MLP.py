import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        mid_features = 32
        self.featurizer = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=mid_features,
            ),
        )
        self.featurizer.n_outputs = mid_features
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=mid_features,
                out_features=out_features,
            ),
        )

    def forward(self, x):
        f = self.featurizer(x)
        v = self.dense(f)
        return v
