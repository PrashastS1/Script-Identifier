from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from loguru import logger
from typing import Dict, Any
from .BH_scene_dataset import BHSceneDataset
from collections import defaultdict
import json


class PairedLanguageDataset(Dataset):
    def __init__(
            self,
            root_dir: str = "data/recognition", 
            train_split: bool = True, 
            transformation: bool = True,
            backbone: str = None,
            gap_dim: int = 0,
        ) -> None:
        """
        Dataset that creates pairs of images, either from the same language or different languages.
        
        Args:
        - root_dir: str, path to the root directory of the dataset
        - train_split: bool, whether to use train split or test split
        - transformation: bool, whether to use albumentations for transformations
        - backbone: str, backbone to be used for feature extraction (affects normalization)
        - gap_dim: int, dimension of the global average pooled features
        
        NOTE:
        - run code as python -m dataset.BH_scene_dataset_concatenate to test the dataset
        - run any file from the root directory of the project as module (python -m <module_name/file_name>)
        """
        
        super(PairedLanguageDataset, self).__init__()
        self.root_dir = root_dir
        self.train_split = train_split
        self.transformation = transformation
        self.backbone_name = backbone
        self.gap_dim = gap_dim
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_dataset = BHSceneDataset(
            root_dir=self.root_dir,
            train_split=self.train_split,
            transformation=False,
            backbone=self.backbone_name,
            gap_dim=self.gap_dim
        )

        # Load dataset
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)
        
        # Load language mapping
        with open('./dataset/language_encode.json') as f:
            self.language_mapping = json.load(f)
        
        # Encode languages
        self.csv['Language_id'] = self.csv['Language'].apply(lambda x: self.encode_language(x))
        
        # Group images by language for efficient sampling
        # self.language_groups = {}
        # for lang in self.csv['Language'].unique():
        #     self.language_groups[lang] = self.csv[self.csv['Language'] == lang].index.tolist()

        logger.info(f"Number of images in dataset: {len(self.csv)}")
        self.class_indices = defaultdict(list)
        for idx, lang_id in enumerate(self.csv['Language_id']):
            self.class_indices[lang_id].append(idx)

        self.classes = list(self.class_indices.keys())
        logger.info(f"Number of classes: {len(self.classes)}")

    def encode_language(self, language: str):
        if language not in self.language_mapping:
            raise ValueError(f"Language {language} not in mapping")
        return self.language_mapping[language]

    # def _create_pairs(self) -> List[Tuple[int, int, bool]]:
    #     """
    #     Creates pairs of image indices along with a flag indicating if they're the same language.
    #     Returns a list of tuples (idx1, idx2, is_same_language)
    #     """
    #     pairs = []
    #     total_samples = len(self.csv)
        
    #     # Calculate how many pairs of each type to create
    #     num_pairs = total_samples  # Create as many pairs as there are original images
    #     num_same_lang = int(num_pairs * self.same_lang_ratio)
    #     num_diff_lang = num_pairs - num_same_lang
        
    #     # Create same-language pairs
    #     for _ in range(num_same_lang):
    #         # Randomly select a language
    #         lang = random.choice(list(self.language_groups.keys()))
    #         # If this language has at least 2 images
    #         if len(self.language_groups[lang]) >= 2:
    #             idx1, idx2 = random.sample(self.language_groups[lang], 2)
    #             pairs.append((idx1, idx2, True))
    #         else:
    #             # If not enough images, just duplicate the same image
    #             idx = random.choice(self.language_groups[lang])
    #             pairs.append((idx, idx, True))
        
    #     # Create different-language pairs
    #     for _ in range(num_diff_lang):
    #         # Select two different languages
    #         lang1, lang2 = random.sample(list(self.language_groups.keys()), 2)
    #         idx1 = random.choice(self.language_groups[lang1])
    #         idx2 = random.choice(self.language_groups[lang2])
    #         pairs.append((idx1, idx2, False))
        
    #     # Shuffle the pairs
    #     random.shuffle(pairs)
    #     return pairs

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        anchor_img, anchor_lang = self.base_dataset[index]

        if np.random.rand() < 0.5:
            # Positive pair (same class)
            target_class = anchor_lang
            same_class = True
        else:
            # Negative pair (different class)
            target_class = np.random.choice([c for c in self.classes if c != anchor_lang])
            same_class = False
        
        target_indices = self.class_indices[target_class]
        pair_index = np.random.choice(target_indices)

        while same_class and pair_index == index:
            pair_index = np.random.choice(target_indices)
        
        pair_img, _ = self.base_dataset[pair_index]

        # logger.debug(f"Anchor language: {anchor_lang}, Target language: {target_class}")
        # logger.debug(f"Anchor index: {index}, Pair index: {pair_index}")
        # logger.debug(f"Anchor image shape: {anchor_img.shape}, Pair image shape: {pair_img.shape}")

        ## Concatenate images
        assert anchor_img.shape == pair_img.shape, "Image channels do not match"

        return torch.cat((anchor_img, pair_img), dim=0), np.float32(1 if same_class else 0)


# Example usage
if __name__ == "__main__":
    dataset = PairedLanguageDataset(
        root_dir="data/recognition",
        train_split=True,
        transformation=True,
        backbone="resnet50",
        gap_dim=1
    )
    
    # Test the dataset
    for i in range(5):
        img, label = dataset[i]
        print(f"Image shape: {img.shape}, Label: {label}")
