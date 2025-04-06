from torchvision import models
import torch
import torch.nn as nn
from loguru import logger


class RESNET_backbone(nn.Module):
    def __init__(self, pretrained: bool = True, gap_dim: int=1):
        super(RESNET_backbone, self).__init__()
        if pretrained:
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        else:
            raise NotImplementedError("RESNET50 model is only available with pretrained weights")
        logger.info("RESNET50 model loaded")
        self.model = nn.Sequential(*list(model.children())[:-2])
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All RESNET parameters are frozen.")
        if gap_dim > 7:
            raise ValueError("Global Average Pooling dimension should be less than 7")
        self.gap = nn.AdaptiveAvgPool2d((gap_dim, gap_dim))
        logger.info("RESNET50 model modified")

    def forward(self, x):
        ## output is 2048x7x7
        x = self.model(x)
        return self.gap(x).view(x.size(0), -1)

if __name__ == "__main__":
    model = RESNET_backbone(pretrained=True, gap_dim=1)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
