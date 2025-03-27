import torch.nn as nn
from loguru import logger
from typing_extensions import Dict


class ANN_base(nn.Module):
    def __init__(self, config: Dict):
        super(ANN_base, self).__init__()
        self.num_feature = config["num_feature"]
        self.num_class = config["num_class"]
        self.model = nn.Linear(self.num_feature, self.num_class, bias=True)
        logger.info("ANN_base initialized")

    def forward(self, x):
        return self.model(x)
