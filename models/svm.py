import numpy as np
from collections import Counter
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from models.backbones.resnet50 import RESNET_backbone
from models.backbones.vgg import VGG_backbone
from models.backbones.vit import VIT_backbone
from loguru import logger
from dataset.BH_scene_dataset import BHSceneDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# List of languages from the image
languages = [
    'assamese', 'bengali', 'english', 'gujarati', 'hindi', 
    'kannada', 'malayalam', 'marathi', 'meitei', 'odia', 
    'punjabi', 'tamil', 'telugu', 'urdu'
]

# Create a LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(languages)

class SVM:
    def __init__(self, model_name, backbone, num_classes, device):
        self.model_name = model_name
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def load_data(self, data_path):
        dataset = BHSceneDataset(data_path)
        return dataset

    def load_model(self):
        if self.backbone == 'resnet50':
            self.model = RESNET_backbone(self.num_classes)
        elif self.backbone == 'vgg':
            self.model = VGG_backbone(self.num_classes)
        elif self.backbone == 'vit':
            self.model = VIT_backbone(self.num_classes)
        else:
            logger.error('Invalid backbone model')
            return

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    def train(self, data_path, batch_size, epochs):
        dataset = self.load_data(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.load_model()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0

            for data in tqdm(dataloader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += self.get_num_correct(outputs, labels)

            self.scheduler.step(total_loss)
            logger.info(f'Epoch: {epoch}, Loss: {total_loss}, Accuracy: {total_correct / len(dataset)}')

    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def predict(self, image):
        self.model.eval()
        image = image.to(self.device)
        outputs = self.model(image)
        _, preds = torch.max(outputs, 1)
        return label_encoder.inverse_transform(preds.cpu().numpy())
    