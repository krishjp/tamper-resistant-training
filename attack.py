from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from PIL import Image

from attack_profiles import pgd_attack
from utils.image_display import plot_image_comparison

def resnet_attack(debug=False):
    model_name = "microsoft/resnet-18"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True) 
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()

    labels = model.config.id2label

    cat_image = Image.open('data/input/tabby_cat.jpg').convert('RGB')

    original_label_index = 281 #tabby cat
    target_label_index = 207 # golden retriever

    print(f"Original Class Goal: {labels[original_label_index]}")
    print(f"Adversarial Target: {labels[target_label_index]}")

    inputs_clean = processor(images=cat_image, return_tensors="pt")
    with torch.no_grad():
        outputs_clean = model(**inputs_clean)
    predicted_class_idx_clean = outputs_clean.logits.argmax(-1).item()
    print(f"\nOriginal Prediction: {labels[predicted_class_idx_clean]}")

    adversarial_pixels = pgd_attack(model, processor, cat_image, target_label_index)

    if debug:
        plot_image_comparison(
            images=[cat_image, adversarial_pixels],
            titles=["Original Image", "Adversarial Image"],
            save_path='data/output/nesnet_cat_compare.png',
            size=(224, 224),
        )

    with torch.no_grad():
        outputs_adv = model(pixel_values=adversarial_pixels)

    predicted_class_idx_adv = outputs_adv.logits.argmax(-1).item()
    print(f"Adversarial Prediction: {labels[predicted_class_idx_adv]}")

if __name__ == "__main__":
    resnet_attack(debug=False)
