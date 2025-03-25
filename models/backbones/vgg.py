from torchvision import models
import torch
import torch.nn as nn
from loguru import logger


class VGG_backbone(nn.Module):
    def __init__(self, pretrained: bool = True, gap_dim: int=1):
        super(VGG_backbone, self).__init__()
        if pretrained:
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        else:
            raise NotImplementedError("VGG16 model is only available with pretrained weights")
        logger.info("VGG16 model loaded")
        self.model = nn.Sequential(*list(model.features.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All VGG16 parameters are frozen.")
        if gap_dim > 14:
            raise ValueError("Global Average Pooling dimension should be less than 14")
        self.gap = nn.AdaptiveAvgPool2d((gap_dim, gap_dim))
        logger.info("VGG16 model modified")

    def forward(self, x):
        ## output is 512x14x14
        x = self.model(x)
        return self.gap(x).view(x.size(0), -1)

if __name__ == "__main__":
    model = VGG_backbone(pretrained=True, gap_dim=2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
