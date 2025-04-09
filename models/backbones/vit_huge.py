from torchvision import models
import torch
import torch.nn as nn
from loguru import logger


class VIT_huge_backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(VIT_huge_backbone, self).__init__()
        if pretrained:
            import timm
            self.model = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True)
        else:
            raise NotImplementedError("ViT model is only available with pretrained weights")
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Access timm model components directly
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm

    def forward(self, x):
        # Timm's ViT processing
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Forward through transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x[:, 0]  # Return class token


if __name__ == "__main__":
    model = VIT_huge_backbone(pretrained=True)
    print('got it')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
