from torchvision import models
import torch
import torch.nn as nn
from loguru import logger


class VIT_LARGE_backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(VIT_LARGE_backbone, self).__init__()
        if pretrained:
            model = models.vit_l_16(weights='ViT_L_16_Weights.DEFAULT')
        else:
            raise NotImplementedError("ViT model is only available with pretrained weights")
        logger.info("ViT model loaded")
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("All VIT parameters are frozen.")
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.conv = feature_extractor[0]
        self.encoder = feature_extractor[1]

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


if __name__ == "__main__":
    model = VIT_LARGE_backbone(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
