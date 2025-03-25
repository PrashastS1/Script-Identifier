from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np
from loguru import logger
from typing import Dict, Any
from models.backbones.resnet50 import RESNET_backbone
from models.backbones.vgg import VGG_backbone
from models.backbones.vit import VIT_backbone

class BHSceneDataset(Dataset):
    def __init__(
            self, 
            root_dir: str = "data/recognition", 
            train_split: bool = True, 
            transform=None,
            linear_transform: bool = False,
            backbone: str = None,
            gap_dim: int = 1
        ) -> None:

        """
        Args:
        - root_dir: str, path to the root directory of the dataset
        - train_split: bool, whether to use train split or test split
        - transform: albumentations.Compose, albumentations transform to be applied to the images
        - linear_transform: bool, whether to linearize the image before passing to the backbone
        - backbone: str, backbone to be used for feature extraction
        - gap_dim: int, dimension of the global average pooled features

        NOTE:
        - run code as python -m dataset.BH_scene_dataset to test the dataset
        - run any file from the root directory of the project as module (python -m <module_name/file_name>)
        """
        
        super(BHSceneDataset, self).__init__()
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.transform = transform
        self.linear_transform = linear_transform

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"csv file not present at {self.csv_path}")
        
        if backbone is None:
            self.backbone = None
            logger.info("No backbone specified")
        elif backbone == 'resnet50':
            self.backbone = RESNET_backbone(pretrained=True, gap_dim=gap_dim)
            logger.info("Using ResNet50 backbone")
        elif backbone == 'vgg':
            self.backbone = VGG_backbone(pretrained=True, gap_dim=gap_dim)
            logger.info("Using VGG backbone")
        elif backbone == 'vit':
            self.backbone = VIT_backbone(pretrained=True)
            logger.info("Using VIT backbone")
            logger.warning("gap_dim is not used for ViT backbone")
        else:
            raise ValueError(f"Invalid backbone: {backbone}, valid backbones are: resnet50, vgg, vit")
            
        
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)
        logger.info(f"Loaded csv file from {self.csv_path}")
        logger.info(f"Dataset formed with {len(self.csv)} samples")

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self.csv.iloc[index]
        
        img_path = os.path.join(self.root_dir, row['Filepath'])
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            ## if image size is hxw, make it sxs , where s=max(h,w)
            h, w = image.shape[:2]
            s = max(h, w)
            ## add req padding
            top_padding = (s-h)//2
            bottom_padding = (s-h) - top_padding
            left_padding = (s-w)//2
            right_padding = (s-w) - left_padding
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            image = cv2.resize(image, (224, 224))

        if self.backbone is not None:
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
            assert image.shape == (1, 3, 224, 224), f"Image shape is {image.shape}"
            image = self.backbone(image)
            # image = image.squeeze(0)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float()
            if self.linear_transform:
                # current dim - 1x3x224x224
                image = image.reshape(-1)
            
        return {
            'image': image,
            'text': row['Text'],
            'language': row['Language']
        }


if __name__ == "__main__":
    dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=True,
        transform=None,
        linear_transform=True,
        backbone='resnet50',
        gap_dim=1
    )
    print(len(dataset))
    for i in range(10):
        print(dataset[i]['image'].shape)
        print(dataset[i]['language'])
