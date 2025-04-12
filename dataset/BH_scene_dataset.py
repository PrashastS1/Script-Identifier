from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np
from loguru import logger
from skimage.feature import (
    hog,
    local_binary_pattern
)
from typing import Dict, Any
from models.backbones.resnet50 import RESNET_backbone
from models.backbones.vgg import VGG_backbone
from models.backbones.vit import VIT_backbone
from models.backbones.vit_large import VIT_LARGE_backbone
from .transformations import LanguageRecognitionTransforms
from tqdm import tqdm
import json
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

class BHSceneDataset(Dataset):
    def __init__(
            self,
            root_dir: str = "data/recognition", 
            train_split: bool = True, 
            transformation: bool = True,
            backbone: str = None,
            gap_dim: int = 1
        ) -> None:

        """
        Args:
        - root_dir: str, path to the root directory of the dataset
        - train_split: bool, whether to use train split or test split
        - Transformation: bool, whether to use albumentations for data augmentation
        #removed (not needed)  - linear_transform: bool, whether to linearize the image before passing to the backbone
        - backbone: str, backbone to be used for feature extraction ## resnet50, vgg, vit, vit_large, hog, sift, lbp
        ###### swin, beit in progress
        - gap_dim: int, dimension of the global average pooled features

        NOTE:
        - run code as python -m dataset.BH_scene_dataset to test the dataset
        - run any file from the root directory of the project as module (python -m <module_name/file_name>)
        """
        
        super(BHSceneDataset, self).__init__()
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.linear_transform = True
        self.backbone_name = backbone
        self.backbone = backbone
        self.gap_dim = gap_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.linear_transform:
            logger.warning("linear_transform is set to False")

        if self.backbone is not None and self.linear_transform is False:
            logger.warning("Backbone is specified but linear_transform is False, setting linear_transform to True")
            self.linear_transform = True

        if self.gap_dim and self.backbone is None:
            logger.warning("gap_dim is specified but backbone is None")
        
        if not self.gap_dim and self.backbone and self.backbone != 'vit' and self.backbone != 'hog' and self.backbone != 'sift' and self.backbone != 'vit_large' and self.backbone != 'lbp':
            logger.warning(f"gap_dim is not specified but backbone is not {self.backbone}, setting gap_dim to 1")
            self.gap_dim = 1

        if self.gap_dim and self.backbone in ['vit', 'vit_large', 'hog', 'sift', 'lbp']:
            logger.warning(f"gap_dim does not matter for {self.backbone}")
            self.gap_dim = 1

        if self.backbone in ['vit', 'vit_large', 'sift', 'hog', 'lbp']:
            self.gap_dim = 1 

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"csv file not present at {self.csv_path}")
        
        self.latent_dir = os.path.join(
            "data",
            f"latent_{backbone}_{'data_augmentation' if transformation else 'no_data_augmentation'}_{self.gap_dim}",
            "train" if train_split else "test"
        )

        if not os.path.exists(self.latent_dir):
            os.makedirs(self.latent_dir)
            logger.info(f"latent_dir not present")
            logger.info(f"Created directory {self.latent_dir}")
        else:
            logger.info(f"latent_dir already present at {self.latent_dir}")

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
        elif backbone == 'vit_large':
            self.backbone = VIT_LARGE_backbone(pretrained=True).to(self.device)
            logger.info("Using VIT_large backbone")
        elif backbone == "sift":
            self.backbone = cv2.SIFT_create()
            self.topk = 64
            self.expected_dim = 128*self.topk
            logger.info("Using SIFT for feature extraction")
        elif backbone == "hog":
            logger.info(f"Using HOG for feature extraction")
        elif backbone == "lbp":
            logger.info(f"Using LBP for feature extraction")
        else:
            raise ValueError(f"Invalid backbone: {backbone}, use valid backbone - resnet50, vgg, vit, vit_large, sift, hog")
        
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
    
    def apply_sift(self, image):
        np_image = image.squeeze(0).permute(1, 2, 0).cpu()
        ## print max and min element
        # print(np_image.max(), np_image.min())
        # np_image = (image * 255).astype(np.uint8)
        np_image = np_image.numpy().astype(np.uint8)
        # Convert to grayscale (OpenCV expects HxWxC)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.backbone.detectAndCompute(gray, None)
        sorted_indices = np.argsort([-kp.response for kp in keypoints])
        # Select max (len(keypoints), self.topk) indices
        # logger.info(f"total extracted keypoints - {len(sorted_indices)}")
        top_indices = sorted_indices[:self.topk]
        # Select corresponding descriptors
        if len(top_indices) > 0:
            selected_descriptors = descriptors[top_indices]
        else:
            selected_descriptors = np.array([])
        # Flatten the descriptors and return
        selected_descriptors = selected_descriptors.flatten()
        ## if dim < expected_dim, pad with zeros
        if selected_descriptors.shape[0] < self.expected_dim:
            selected_descriptors = np.pad(selected_descriptors, (0, self.expected_dim - selected_descriptors.shape[0]), 'constant')
        return selected_descriptors
    
    def apply_hog(self, image):
        np_image = image.squeeze(0).permute(1, 2, 0).cpu()
        ## print max and min element
        # print(np_image.max(), np_image.min())
        # np_image = (image * 255).astype(np.uint8)
        np_image = np_image.numpy().astype(np.uint8)
        # Convert to grayscale (OpenCV expects HxWxC)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        features = hog(
            gray,
            orientations = 9,
            pixels_per_cell = (8, 8),
            cells_per_block = (2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        return features
    
    def apply_lbp(self, image):
        radii=[1,2,3,4,5,6,7,8,9,10,11,12]
        np_image = image.squeeze(0).permute(1, 2, 0).cpu()
        ## print max and min element
        # print(np_image.max(), np_image.min())
        # np_image = (image * 255).astype(np.uint8)
        np_image = np_image.numpy().astype(np.uint8)
        # Convert to grayscale (OpenCV expects HxWxC)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        features = []
        height, width = gray.shape
        regions = [
            gray[:height//2, :],          # Upper half
            gray[height//4:3*height//4, :], # Central half
            gray[height//2:, :]           # Lower half
        ]
        for region in regions:
            for radius in radii:
                n_points = 8 * radius
                lbp = local_binary_pattern(region, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
                features.extend(hist / (hist.sum() + 1e-6))  # Normalized histogram
        
        return np.array(features)



    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self.csv.iloc[index]
        
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
            if self.backbone_name == 'sift':
                image = self.apply_sift(image)
                image = torch.tensor(image).float().to(self.device)
            elif self.backbone_name == 'hog':
                image = self.apply_hog(image)
                image = torch.tensor(image).float().to(self.device)
            elif self.backbone_name == 'lbp':
                image = self.apply_lbp(image)
                image = torch.tensor(image).float().to(self.device)
            else:
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
    backbone_opt = ['vit', 'vit_large']
    train_split_opt = [True, False]
    transformation_opt = [True, False]
    gap_dim_opt = [1, 2, 3]

    for backbone in backbone_opt:
        for train_split in train_split_opt:
            for transformation in transformation_opt:
                for gap_dim in gap_dim_opt:
                    print(f"Testing with backbone: {backbone}, train_split: {train_split}, transformation: {transformation}, gap_dim: {gap_dim}")
                    dataset = BHSceneDataset(
                        root_dir="data/recognition",
                        train_split=train_split,
                        transformation=transformation,
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
        backbone='lbp',
        gap_dim=2
    )

    # for i in range(10):
    #     img, lang = dataset[i]
    #     print(f"Image shape: {img.shape}, Language: {lang}")

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # for i, (img, tar) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing dataset", unit="sample"):
    #     # img, lang = dataset[i]
    #     print(f"Image shape: {img.shape}, Language: {tar}")


    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features(dataset, batch_size=64):
        """Extract features from dataset using GPU acceleration."""
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=False
        )

        X_list, y_list = [], []
        for batch in tqdm(dataloader, desc="Extracting Features"):
            X, y = batch  # Unpack tuple
            X = X.to(device)  # Move to GPU

            X_list.append(X.cpu().numpy())
            y_list.append(y)

        return np.vstack(X_list), np.concatenate(y_list)

    extract_features(dataset)
