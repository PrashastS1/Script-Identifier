from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np
import loguru
from typing import Dict, Any

class BHSceneDataset(Dataset):
    def __init__(self, root_dir: str = "data/recognition", train_split: bool = True, transform=None):
        super(BHSceneDataset, self).__init__()
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir, "train.csv" if train_split else "test.csv")
        self.transform = transform

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"csv file not present at {self.csv_path}")
        
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)

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
            
        return {
            'image': image,
            'text': row['Text'],
            'language': row['Language']
        }

if __name__ == "__main__":
    dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=True,
        transform=None
    )
    print(len(dataset))
    for i in range(10):
        print(dataset[i]['image'].shape)
        print(dataset[i]['language'])
