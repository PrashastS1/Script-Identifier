import torch.nn as nn
from loguru import logger
from typing_extensions import Dict


class ANN_base(nn.Module):
    def __init__(self, config: Dict):
        super(ANN_base, self).__init__()
        self.layer_sizes = config['layers']
        assert self.layer_sizes[-1] == 14, "Last layer must have 14 outputs"
        layers = []
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != 14:  # No batchnorm/relu for final layer
                layers.append(nn.BatchNorm1d(out_size))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        logger.info("ANN_base initialized")

    def forward(self, x):
        return self.net(x)
