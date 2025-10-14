import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils.loader import get_imagenet_splits
from tqdm.auto import tqdm
import torch.nn.functional as F

# Config
ORIGINAL_MODEL_NAME = "microsoft/resnet-18"
HARDENED_MODEL_PATH = "./resnet-18-imagenet-hardened/final"
EPSILON = 8/255

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.xpu.is_available():
    DEVICE = 'xpu'
else:
    DEVICE = 'cpu'
    raise ValueError('Using CPU')

# FGSM attack
def generate_adversarial_examples(model, images, labels):
    """Generates adversarial examples using the FGSM attack."""
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs.logits, labels)

    model.zero_grad()
    loss.backward()

    grad = images.grad.data

    perturbed_image = images + EPSILON * grad.sign()
    
    adv_images = torch.clamp(perturbed_image, 0, 1)
    
    return adv_images

def evaluate_model(model, dataloader, attack_fn=None):
    """Evaluates the model on a given dataloader, with an optional attack."""
    model.eval()
    correct = 0
    total = 0
    top_n = 5

    if not dataloader or len(dataloader) == 0:
        raise ValueError("Warning: Dataloader is empty")
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        labels = batch['label'].to(DEVICE)
        images = batch['pixel_values'].to(DEVICE)
        
        if attack_fn:
            images = attack_fn(model, images, labels)

        with torch.no_grad():
            outputs = model(images)

        _, top_n_predictions = torch.topk(outputs.logits, k=top_n, dim=1)
        labels_reshaped = labels.view(-1, 1)
        is_in_top_n = top_n_predictions == labels_reshaped
        correct += is_in_top_n.any(dim=1).sum().item()
        total += labels.size(0)

        # predictions = torch.argmax(outputs.logits, dim=1)
        # total += labels.size(0)
        # correct += (predictions == labels).sum().item()
        
        progress_bar.set_postfix({"Accuracy": f"{(100 * correct / total):.2f}%"})

    return 100 * correct / total


processor = AutoImageProcessor.from_pretrained(ORIGINAL_MODEL_NAME, use_fast=True)
def transform(examples):
        examples["pixel_values"] = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")['pixel_values']
        return examples

def main():
    print(f"Using device: {DEVICE}")

    original_model = AutoModelForImageClassification.from_pretrained(ORIGINAL_MODEL_NAME).to(DEVICE)
    hardened_model = AutoModelForImageClassification.from_pretrained(HARDENED_MODEL_PATH).to(DEVICE)
    
    split_dataset = get_imagenet_splits()
    val_dataset = split_dataset['validation']
    
    processed_val_dataset = val_dataset.map(transform, batched=True)
    processed_val_dataset = processed_val_dataset.remove_columns('image')

    processed_val_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    val_dataloader = torch.utils.data.DataLoader(processed_val_dataset, batch_size=64)

    acc_orig_clean = evaluate_model(original_model, val_dataloader)
    acc_orig_adv = evaluate_model(original_model, val_dataloader, attack_fn=generate_adversarial_examples)

    acc_hard_clean = evaluate_model(hardened_model, val_dataloader)
    acc_hard_adv = evaluate_model(hardened_model, val_dataloader, attack_fn=generate_adversarial_examples)

    print("\n" + "="*30)
    print("PERFORMANCE SUMMARY (IMAGENET)")
    print("="*30)
    print(f"Original Model (Clean):\t{acc_orig_clean:.2f}%")
    print(f"Original Model (Attacked):\t{acc_orig_adv:.2f}%")
    print("-" * 30)
    print(f"Hardened Model (Clean):\t{acc_hard_clean:.2f}%")
    print(f"Hardened Model (Attacked):\t{acc_hard_adv:.2f}%")
    print("="*30)


if __name__ == "__main__":
    main()