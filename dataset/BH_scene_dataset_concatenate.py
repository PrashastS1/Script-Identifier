from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np
from loguru import logger
import random
from typing import Dict, Any, Tuple, List
from .transformations import LanguageRecognitionTransforms
import json


class PairedLanguageDataset(Dataset):
    def __init__(
            self,
            root_dir: str = "data/recognition", 
            train_split: bool = True, 
            transformation: bool = True,
            backbone: str = None,
            img_size: int = 224,
            same_lang_ratio: float = 0.5,
            concat_axis: int = 1  # 0 for vertical, 1 for horizontal concatenation
        ) -> None:
        """
        Dataset that creates pairs of images, either from the same language or different languages.
        
        Args:
        - root_dir: str, path to the root directory of the dataset
        - train_split: bool, whether to use train split or test split
        - transformation: bool, whether to use albumentations for transformations
        - backbone: str, backbone to be used for feature extraction (affects normalization)
        - img_size: int, size of the images after transformation
        - same_lang_ratio: float, ratio of same-language pairs to generate (0.0-1.0)
        - concat_axis: int, axis along which to concatenate images (0=vertical, 1=horizontal)
        """
        
        super(PairedLanguageDataset, self).__init__()
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.img_size = img_size
        self.same_lang_ratio = same_lang_ratio
        self.concat_axis = concat_axis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not present at {self.csv_path}")
        
        # Setup transformations
        if transformation:
            logger.info("Using albumentations for transformations")
            self.transform = LanguageRecognitionTransforms.get_transforms(
                backbone_type=backbone,
                phase='train' if train_split else 'test',
                img_size=img_size
            )
        else:
            logger.info("Not using albumentations for transformations")
            self.transform = None
        
        # Load dataset
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)
        
        # Load language mapping
        with open('./dataset/language_encode.json') as f:
            self.language_mapping = json.load(f)
        
        # Encode languages
        self.csv['Language'] = self.csv['Language'].apply(lambda x: self.encode_language(x))
        
        # Group images by language for efficient sampling
        self.language_groups = {}
        for lang in self.csv['Language'].unique():
            self.language_groups[lang] = self.csv[self.csv['Language'] == lang].index.tolist()
        
        logger.info(f"Loaded csv file from {self.csv_path}")
        logger.info(f"Dataset formed with {len(self.csv)} samples across {len(self.language_groups)} languages")
        
        # Create pairs of indices
        self.pairs = self._create_pairs()
        logger.info(f"Created {len(self.pairs)} image pairs")

    def encode_language(self, language: str):
        if language not in self.language_mapping:
            raise ValueError(f"Language {language} not in mapping")
        return self.language_mapping[language]

    def _create_pairs(self) -> List[Tuple[int, int, bool]]:
        """
        Creates pairs of image indices along with a flag indicating if they're the same language.
        Returns a list of tuples (idx1, idx2, is_same_language)
        """
        pairs = []
        total_samples = len(self.csv)
        
        # Calculate how many pairs of each type to create
        num_pairs = total_samples  # Create as many pairs as there are original images
        num_same_lang = int(num_pairs * self.same_lang_ratio)
        num_diff_lang = num_pairs - num_same_lang
        
        # Create same-language pairs
        for _ in range(num_same_lang):
            # Randomly select a language
            lang = random.choice(list(self.language_groups.keys()))
            # If this language has at least 2 images
            if len(self.language_groups[lang]) >= 2:
                idx1, idx2 = random.sample(self.language_groups[lang], 2)
                pairs.append((idx1, idx2, True))
            else:
                # If not enough images, just duplicate the same image
                idx = random.choice(self.language_groups[lang])
                pairs.append((idx, idx, True))
        
        # Create different-language pairs
        for _ in range(num_diff_lang):
            # Select two different languages
            lang1, lang2 = random.sample(list(self.language_groups.keys()), 2)
            idx1 = random.choice(self.language_groups[lang1])
            idx2 = random.choice(self.language_groups[lang2])
            pairs.append((idx1, idx2, False))
        
        # Shuffle the pairs
        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def load_and_process_image(self, img_path: str) -> torch.Tensor:
        """Load and process a single image"""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Basic processing if no transform is provided
            h, w = image.shape[:2]
            s = max(h, w)
            # Add padding
            top_padding = (s-h)//2
            bottom_padding = (s-h) - top_padding
            left_padding = (s-w)//2
            right_padding = (s-w) - left_padding
            image = cv2.copyMakeBorder(
                image, top_padding, bottom_padding, left_padding, right_padding, 
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            image = cv2.resize(image, (self.img_size, self.img_size))
            # Convert to tensor format
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        return image

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx1, idx2, is_same_lang = self.pairs[index]
        
        # Get image paths
        img_path1 = os.path.join(self.root_dir, self.csv.iloc[idx1]['Filepath'])
        img_path2 = os.path.join(self.root_dir, self.csv.iloc[idx2]['Filepath'])
        
        # Load and process images
        img1 = self.load_and_process_image(img_path1)
        img2 = self.load_and_process_image(img_path2)
        
        # Get labels
        label1 = self.csv.iloc[idx1]['Language']
        label2 = self.csv.iloc[idx2]['Language']
        
        # Concatenate images
        concat_img = torch.cat([img1, img2], dim=self.concat_axis + 1)  # +1 because first dim is channels
        
        return {
            'image': concat_img,
            'label1': label1,
            'label2': label2,
            'is_same_language': torch.tensor(1.0 if is_same_lang else 0.0),
            'original_indices': (idx1, idx2)
        }


# Example usage
if __name__ == "__main__":
    dataset = PairedLanguageDataset(
        root_dir="data/recognition",
        train_split=True,
        transformation=True,
        backbone='resnet50',
        img_size=224,
        same_lang_ratio=0.5,
        concat_axis=1  # horizontal concatenation
    )
    
    # Test the dataset
    sample = dataset[0]
    print(f"Concatenated image shape: {sample['image'].shape}")
    print(f"Label 1: {sample['label1']}, Label 2: {sample['label2']}")
    print(f"Is same language: {sample['is_same_language']}")
    # To use with DataLoader:
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
