import torch
import torch.nn as nn
from dataset.BH_scene_dataset import BHSceneDataset
from dataset.transformations import LanguageRecognitionTransforms
from models.backbones.vit_large import VIT_LARGE_backbone
from models.ann import ANN_base
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image
import cv2

# Global variables to store loaded model and dataset
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_transformation = None
_backbone = VIT_LARGE_backbone().to(_device).eval()

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
    _transformation = transformation
    return transformation

def process_image(img: Image) -> torch.Tensor:
    """Preprocess image for model input."""
    global _backbone
    logger.info("Processing image...")
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformation = get_test_transformation(backbone_type='vit_large')
    img = transformation(image=img)['image'].unsqueeze(0).float().to(_device)
    logger.info("Image processed.")
    img = _backbone(img)
    logger.info("Image passed through backbone.")
    return img

def load_model_once(model_path: str = "ckpts/model_ann_a1.pt", device: str = "cuda") -> nn.Module:
    """Load model only once."""
    global _model
    if _model is not None:
        return _model
    logger.info("Loading model...")
    model_config = {
        "layers": [1024, 512, 256, 14],
        "dropout": [0.25, 0.3]
    }

    _model = ANN_base(config=model_config)
    _model.load_state_dict(torch.load(model_path, map_location=device))
    _model = _model.to(device)
    _model.eval()
    logger.info("Model loaded successfully.")
    return _model

def predict_class(
    model,
    query_image: torch.Tensor,
):
    """Predict class for query image using Siamese network."""
    model.eval()
    logger.info(f"img shape - {query_image.shape}")
    output_logits = model(query_image)
    ## also return confidence score
    # _, predicted_class = torch.max(output_logits, 1)
    # predicted_class = predicted_class.item()
    predicted_class = torch.argmax(output_logits, dim=1).item()
    probability = torch.softmax(output_logits, dim=1).max().item()
    logger.info(f"Predicted class: {predicted_class}")
    # logger.info(f"Predicted class: {predicted_class}")
    return predicted_class, probability


def test():
    test_dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=False,
        transformation=True,
        backbone='vit_large',
        gap_dim=1
    )
    expected_out = []
    actual_out = []


    # Load model
    load_model_once(model_path = "ckpts/model_ann_a1.pt")
    
    for i in tqdm(range(len(test_dataset)), desc="Testing"):
        img, label = test_dataset[i]
        img.unsqueeze_(0)
        expected_out.append(label.item())
        # Predict class
        predicted_class, _ = predict_class(load_model_once(), img)
        actual_out.append(predicted_class)

    ## save results
    # np.save("actual_out.npy", actual_out)
    # np.save("expected_out.npy", expected_out)
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


_model = load_model_once(model_path = "ckpts/model_ann_a1.pt")
_transformation = get_test_transformation(backbone_type='vit_large')
_model.eval()

if __name__ == "__main__":
    # Example usage
    test()