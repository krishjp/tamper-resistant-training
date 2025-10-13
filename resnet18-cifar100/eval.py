import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from tqdm.auto import tqdm
import torch.nn.functional as F

ORIGINAL_MODEL_NAME = "edadaltocg/resnet18_cifar100"
HARDENED_MODEL_PATH = "./resnet-18-cifar100-hardened/final"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPSILON = 8/255

def generate_adversarial_examples(model, images, labels):
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
    model.eval()
    correct = 0
    total = 0
    
    if not dataloader or len(dataloader) == 0:
        raise ValueError("Warning: Dataloader is empty")
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        labels = batch['labels'].to(DEVICE)
        images = batch['pixel_values'].to(DEVICE)
        
        if attack_fn:
            images = attack_fn(model, images, labels)

        with torch.no_grad():
            outputs = model(images)

        predictions = torch.argmax(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
        progress_bar.set_postfix({"Accuracy": f"{(100 * correct / total):.2f}%"})

    return 100 * correct / total

processor = AutoImageProcessor.from_pretrained(ORIGINAL_MODEL_NAME, use_fast=True)
def transform(examples):
    examples["pixel_values"] = processor([img.convert("RGB") for img in examples["img"]], return_tensors="pt")['pixel_values']
    return examples

def main():
    print(f"Using device: {DEVICE}")

    original_model = AutoModelForImageClassification.from_pretrained(
        ORIGINAL_MODEL_NAME,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    hardened_model = AutoModelForImageClassification.from_pretrained(
        HARDENED_MODEL_PATH,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    split_dataset = load_dataset("cifar100")
    val_dataset = split_dataset['test'] # Use the test set for final evaluation
    
    processed_val_dataset = val_dataset.map(transform, batched=True)
    processed_val_dataset = processed_val_dataset.remove_columns('img')
    processed_val_dataset = processed_val_dataset.rename_column("fine_label", "labels")
    processed_val_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    
    val_dataloader = torch.utils.data.DataLoader(processed_val_dataset, batch_size=64)

    print("\n--- Evaluating Original Model ---")
    acc_orig_clean = evaluate_model(original_model, val_dataloader)
    acc_orig_adv = evaluate_model(original_model, val_dataloader, attack_fn=generate_adversarial_examples)

    print("\n--- Evaluating Hardened Model ---")
    acc_hard_clean = evaluate_model(hardened_model, val_dataloader)
    acc_hard_adv = evaluate_model(hardened_model, val_dataloader, attack_fn=generate_adversarial_examples)

    print("\n" + "="*30)
    print(" PERFORMANCE SUMMARY (CIFAR-100)")
    print("="*30)
    print(f"Original Model (Clean):\t{acc_orig_clean:.2f}%")
    print(f"Original Model (Attacked):\t{acc_orig_adv:.2f}%")
    print("-" * 30)
    print(f"Hardened Model (Clean):\t{acc_hard_clean:.2f}%")
    print(f"Hardened Model (Attacked):\t{acc_hard_adv:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()