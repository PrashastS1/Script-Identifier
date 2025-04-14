from PIL import Image
from tools.inference_a1 import(
    load_model_once,
    predict_class,
    process_image
)
from loguru import logger
import time
import json

def best_model(image: Image):
    time.sleep(2)
    logger.info("image processing")
    image = process_image(img=image)

    predicted_cls, confidence = predict_class(load_model_once(), image)
    logger.info(f"Predicted class: {predicted_cls}, Confidence: {confidence}")
    with open('dataset/language_encode.json', 'r') as f:
        language_encode = json.load(f)
    # predicted_language = language_encode.get(predicted_cls, "other")
    ## predicted_language = language encode key with value = predicted_cls
    predicted_language = list(language_encode.keys())[list(language_encode.values()).index(predicted_cls)]
    logger.info(f"Predicted language: {predicted_language}")
    return predicted_language, confidence*100
