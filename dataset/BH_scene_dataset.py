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
from .transformations import LanguageRecognitionTransforms
import json


class BHSceneDataset(Dataset):
    def __init__(
            self,
            root_dir: str = "data/recognition", 
            train_split: bool = True, 
            transformation: bool = True,
            linear_transform: bool = False,
            backbone: str = None,
            gap_dim: int = 1
        ) -> None:

        """
        Args:
        - root_dir: str, path to the root directory of the dataset
        - train_split: bool, whether to use train split or test split
        - Transformation: bool, whether to use albumentations for transformations
        - linear_transform: bool, whether to linearize the image before passing to the backbone
        - backbone: str, backbone to be used for feature extraction ## resnet50, vgg, vit
        ###### swin, beit in progress
        - gap_dim: int, dimension of the global average pooled features

        NOTE:
        - run code as python -m dataset.BH_scene_dataset to test the dataset
        - run any file from the root directory of the project as module (python -m <module_name/file_name>)
        """
        
        super(BHSceneDataset, self).__init__()
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.linear_transform = linear_transform
        self.backbone = backbone
        self.gap_dim = gap_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dir = os.path.join(
            "data",
            f"latent_{backbone}_{'data_augmentation' if transformation else 'no_data_augmentation'}_{'linear_transform' if linear_transform else 'no_linear_transform'}_{gap_dim}",
            "train" if train_split else "test"
        )

        if not os.path.exists(self.latent_dir):
            os.makedirs(self.latent_dir)
            logger.info(f"latent_dir not present")
            logger.info(f"Created directory {self.latent_dir}")
        else:
            logger.info(f"latent_dir already present at {self.latent_dir}")

        if not self.linear_transform:
            logger.warning("linear_transform is set to False")

        if self.backbone is not None and self.linear_transform is False:
            logger.warning("Backbone is specified but linear_transform is False, setting linear_transform to True")
            self.linear_transform = True

        if self.gap_dim and self.backbone is None:
            logger.warning("gap_dim is specified but backbone is None")
        
        if not self.gap_dim and self.backbone and self.backbone != 'vit':
            logger.warning("gap_dim is not specified but backbone is not vit, setting gap_dim to 1")
            self.gap_dim = 1

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"csv file not present at {self.csv_path}")
        
        if transformation:
            logger.info("Using albumentations for transformations")
            self.transform = LanguageRecognitionTransforms.get_transforms(
                backbone_type=backbone,
                phase='train' if train_split else 'test',
                img_size=224
            )
        else:
            logger.info("Not using albumentations for transformations")
            self.transform = None
        
        if backbone is None:
            self.backbone = None
            logger.info("No backbone specified")
        elif backbone == 'resnet50':
            self.backbone = RESNET_backbone(pretrained=True, gap_dim=gap_dim).to(self.device)
            logger.info("Using ResNet50 backbone")
        elif backbone == 'vgg':
            self.backbone = VGG_backbone(pretrained=True, gap_dim=gap_dim).to(self.device)
            logger.info("Using VGG backbone")
        elif backbone == 'vit':
            self.backbone = VIT_backbone(pretrained=True).to(self.device)
            logger.info("Using VIT backbone")
        else:
            raise ValueError(f"Invalid backbone: {backbone}, valid backbones are: resnet50, vgg, vit")
            
        
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)

        with open('./dataset/language_encode.json') as f:
            self.language_mapping = json.load(f)

        self.csv['Language_id'] = self.csv['Language'].apply(lambda x : self.encode_language(x))
        ## print unique language
        # print(self.csv['Language'].unique())
        logger.info(f"Loaded csv file from {self.csv_path}")
        logger.info(f"Dataset formed with {len(self.csv)} samples")

    def encode_language(self, language: str):
        # if language is not in the mapping, raise an error
        if language not in self.language_mapping:
            raise ValueError(f"Language {language} not in mapping")
        # if language is in the mapping, return the corresponding value
        return self.language_mapping[language]

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        
        ## latent dir
        latent_image_dir = os.path.join(
            self.latent_dir,
            row['Language']
        )

        if not os.path.exists(
            latent_image_dir
        ):
            os.makedirs(latent_image_dir)
            logger.info(f"Created directory {latent_image_dir}")

        ## save latent features
        latent_image_path = os.path.join(
            latent_image_dir,
            os.path.basename(row['Filepath']).split('.')[0] + '.npy'
        )

        if os.path.exists(latent_image_path):
            latent = np.load(latent_image_path)
            return torch.tensor(latent).float().to(self.device), row['Language_id']

        row = self.csv.iloc[index]
        
        img_path = os.path.join(self.root_dir, row['Filepath'])
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image=image)['image'].unsqueeze(0).float().to(self.device)
            ## output dim - 3x224x224
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
            ## output dim - 224x224x3
            ## convert to 3x224x224
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        ## for transformations, image is already in the required format
        if self.backbone is not None:
            assert image.shape == (1, 3, 224, 224), f"Image shape is {image.shape}"
            image = self.backbone(image)
            image = image.squeeze(0)
        else:
            if self.linear_transform:
                # current dim - 1x3x224x224
                image = image.reshape(-1)

        ## save latent features at latent_image_path
        np.save(latent_image_path, image.cpu().detach().numpy())
    
        return image, row['Language_id']


def test_dataset():
    ## test for all possbile value
    backbone_opt = ['resnet50', 'vgg', 'vit', None]
    train_split_opt = [True, False]
    transformation_opt = [True, False]
    linear_transform_opt = [True, False]
    gap_dim_opt = [1, 2, 3]

    for backbone in backbone_opt:
        for train_split in train_split_opt:
            for transformation in transformation_opt:
                for linear_transform in linear_transform_opt:
                    for gap_dim in gap_dim_opt:
                        print(f"Testing with backbone: {backbone}, train_split: {train_split}, transformation: {transformation}, linear_transform: {linear_transform}, gap_dim: {gap_dim}")
                        dataset = BHSceneDataset(
                            root_dir="data/recognition",
                            train_split=train_split,
                            transformation=transformation,
                            linear_transform=linear_transform,
                            backbone=backbone,
                            gap_dim=gap_dim
                        )
                        for i in range(1):
                            img, lang = dataset[i]
                            print(f"Image shape: {img.shape}, Language: {lang}")
                        
                        print("\n" + "="*50 + "\n")

                        del dataset
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    # test_dataset()

    dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=False,
        transformation=True,
        linear_transform=False,
        backbone='resnet50',
        gap_dim=1
    )

    for i in range(1):
        img, lang = dataset[i]
        print(f"Image shape: {img.shape}, Language: {lang}")
