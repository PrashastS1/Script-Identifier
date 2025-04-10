import torch
import torch.nn as nn
from dataset.BH_scene_dataset import BHSceneDataset
from dataset.transformations import LanguageRecognitionTransforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.backbones.vit_huge import VIT_huge_backbone
from models.pair_ann import PAIR_ANN
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image
import cv2

# Global variables to store loaded model and dataset
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_reference_loader = None
_model = None
_transformation = None
_backbone = VIT_huge_backbone().to(_device)

def get_test_transformation(
    backbone_type: str,
    img_size: int = 224,
) :
    global _transformation
    if _transformation is not None:
        return _transformation
    logger.info("Creating test transformation...")
    transformation = LanguageRecognitionTransforms.get_transforms(
        backbone_type=backbone_type,
        phase='test',
        img_size=img_size
    )
    logger.info("Test transformation created.")
    return transformation

def process_image(img: Image) -> torch.Tensor:
    """Preprocess image for model input."""
    global _backbone
    logger.info("Processing image...")
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformation = get_test_transformation(backbone_type='vit_huge')
    img = transformation(image=img)['image'].unsqueeze(0).float().to(_device)
    logger.info("Image processed.")
    img = _backbone(img)
    logger.info("Image passed through backbone.")
    return img

def load_reference_dataset_once(
    root_dir: str = "data/recognition",
    transformation: bool = True,
    backbone: str = None,
    samples_per_class: int = 100
) -> DataLoader:
    """Load reference dataset only once."""
    global _reference_loader
    if _reference_loader is not None:
        return _reference_loader

    logger.info("Loading reference dataset...")
    dataset = BHSceneDataset(
        root_dir=root_dir,
        train_split=False,  # Typically use test set for reference
        transformation=transformation,
        backbone=backbone,
        gap_dim=1
    )
    logger.info("Reference dataset loaded.")
    
    # Create balanced subset indices
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        cls = dataset.csv.iloc[idx]['Language']
        class_indices.setdefault(cls, []).append(idx)
    
    selected_indices = []
    for cls, indices in class_indices.items():
        if len(indices) < samples_per_class:
            # raise ValueError(f"Class {cls} has only {len(indices)} samples")
            logger.warning(f"Class {cls} has only {len(indices)} samples. Using all available samples.")
        selected_indices.extend(indices[:samples_per_class])
    
    _reference_loader = DataLoader(
        dataset,
        batch_size=2048,
        sampler=SubsetRandomSampler(selected_indices)
    )
    logger.info("Reference DataLoader created.")
    return _reference_loader


def load_model_once(model_path: str, device: str = "cuda") -> nn.Module:
    """Load model only once."""
    global _model
    if _model is not None:
        return _model

    logger.info("Loading model...")
    model_config = {
        'layers': [2560, 1024, 512, 1]
    }

    _model = PAIR_ANN(config=model_config)
    model_path = "plots/ann_pair_/vit_huge_pair_ann_fin_hope/vit_huge_pair_ann_fin_hope_0.0005_64.pt"
    _model.load_state_dict(torch.load(model_path, map_location=device))
    _model = _model.to(device)
    _model.eval()
    logger.info("Model loaded successfully.")
    return _model

def predict_class(
    model: nn.Module,
    query_image: torch.Tensor,
    reference_loader: DataLoader,
    device: str = "cuda"
) -> str:
    """Predict class for query image using Siamese network."""
    model.eval()
    similarities = {}
    counts = {}
    
    with torch.no_grad():
        for ref_batch, labels in tqdm(reference_loader, desc="Processing references"):
            # Move data to device
            ref_batch = ref_batch.to(device)
            query_batch = query_image.repeat(ref_batch.size(0), 1)
            # logger.info("Query batch shape: {}".format(query_batch.shape))
            # logger.info("Reference batch shape: {}".format(ref_batch.shape))
            # Concatenate along channel dimension
            concatenated = torch.cat([query_batch, ref_batch], dim=1)
            
            # Get predictions
            outputs = torch.sigmoid(model(concatenated)).squeeze()
            
            # Aggregate results
            for prob, label in zip(outputs.cpu().numpy(), labels):
                cls = reference_loader.dataset.csv.iloc[label.item()]['Language_id']
                similarities[cls] = similarities.get(cls, 0.0) + prob.item()
                counts[cls] = counts.get(cls, 0) + 1
    
    # Calculate average probabilities
    avg_probs = {cls: similarities[cls]/counts[cls] for cls in similarities}
    return max(avg_probs, key=avg_probs.get)


def test():
    test_dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=False,
        transformation=True,
        backbone='vit_huge',
        gap_dim=1
    )
    expected_out = []
    actual_out = []


    # Load model
    load_model_once(model_path="plots/ann_pair_/vit_huge_pair_ann_fin_hope_tuned/vit_huge_pair_ann_fin_hope_tuned_0.0005_256.pt")
    load_reference_dataset_once(
        root_dir="data/recognition",
        transformation=True,
        backbone='vit_huge',
        samples_per_class=100
    )

    
    for i in tqdm(range(len(test_dataset)), desc="Testing"):
        img, label = test_dataset[i]
        expected_out.append(label.item())
        # Predict class
        predicted_class = predict_class(_model, img, _reference_loader)
        actual_out.append(predicted_class)

    ## save results
    np.save("actual_out.npy", actual_out)
    np.save("expected_out.npy", expected_out)
    logger.info("Test completed.")

    # Calculate accuracy
    accuracy = np.mean(np.array(expected_out) == np.array(actual_out))
    logger.info(f"Accuracy: {accuracy:.4f}")

    ## f1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(expected_out, actual_out, average='macro')
    logger.info(f"F1 Score: {f1:.4f}")

    ## calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = confusion_matrix(expected_out, actual_out)
    cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()
    logger.info("Confusion matrix saved as confusion_matrix.png")


if __name__ == "__main__":
    # Example usage
    test()
    # img_path = "data/recognition/test/english/A_image_73_3.jpg"
    # img = Image.open(img_path)
    # img = process_image(img)
    # print(img.shape)  # Check the shape of the processed image
    # # Load reference dataset
    # ref_loader = load_reference_dataset_once(
    #     root_dir="data/recognition",
    #     transformation=True,
    #     backbone='vit_huge',
    #     samples_per_class=100
    # )
    # # Load model
    # load_model_once(model_path="plots/ann_pair_/vit_huge_pair_ann_fin_hope/vit_huge_pair_ann_fin_hope_0.0005_64.pt")
    # # Predict class
    # predicted_class = predict_class(_model, img, ref_loader)
    # logger.info(f"Predicted class: {predicted_class}")