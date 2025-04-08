import torch
import torch.nn as nn
from typing import List


class PAIR_ANN(nn.Module):
    def __init__(self, config: List[int]):
        """
        Args:
            layer_sizes: List of layer dimensions including input and output
                         Example: [1536, 512, 256, 1] for concatenated vit features
        """
        super(PAIR_ANN, self).__init__()
        self.layer_sizes = config['layers']
        assert self.layer_sizes[-1] == 1, "Output layer must be of size 1 for binary classification"
        layers = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != 1:  # No batchnorm/relu for final layer
                layers.append(nn.BatchNorm1d(out_size))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))
