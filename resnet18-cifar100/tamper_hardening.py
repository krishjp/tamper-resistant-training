import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset

class AdversarialTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs['pixel_values'].clone().detach()
        labels = inputs['labels'].clone().detach()
        
        pixel_values.requires_grad = True
        outputs_clean = model(pixel_values)
        loss_clean = F.cross_entropy(outputs_clean.logits, labels)
        
        grad = torch.autograd.grad(loss_clean, pixel_values,
                                   retain_graph=True, create_graph=False)[0]
        
        # FGSM
        epsilon_train = 8/255
        perturbation = epsilon_train * grad.sign()
        adv_pixel_values = torch.clamp(pixel_values.detach() + perturbation, min=0, max=1)

        outputs_adv = model(adv_pixel_values)
        loss_adv = F.cross_entropy(outputs_adv.logits, labels)
        
        loss = (loss_clean + loss_adv) / 2

        return (loss, outputs_adv) if return_outputs else loss

MODEL_NAME = "edadaltocg/resnet18_cifar100"  # Pre-trained ResNet-18 on CIFAR-100

processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

def transform(examples):
    examples["pixel_values"] = processor(
        [img.convert("RGB") for img in examples["img"]], 
        return_tensors="pt"
    )['pixel_values']
    return examples

def tamper_harden_resnet_cifar100():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    
    print("Loading CIFAR-100 dataset...")
    split_dataset = load_dataset("cifar100")
    train_dataset = split_dataset['train']
    
    print("Preprocessing dataset...")
    processed_train_dataset = train_dataset.map(transform, batched=True)
    processed_train_dataset = processed_train_dataset.rename_column("fine_label", "labels")

    is_mps_available = torch.backends.mps.is_available()
    if not is_mps_available:
        print("MPS not available. Exiting.")
        return
    
    print(f"MPS device found. Starting adversarial training on CIFAR-100.")
    
    training_args = TrainingArguments(
        output_dir="./resnet-18-cifar100-hardened",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        logging_steps=100,
        save_strategy="epoch",
        learning_rate=2e-5,
        bf16=is_mps_available,
    )

    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
    )

    print("Starting adversarial training...")
    trainer.train()

    hardened_model_path = "./resnet-18-cifar100-hardened/final"
    trainer.save_model(hardened_model_path)
    print(f"Hardened model saved to {hardened_model_path}")

if __name__ == "__main__":
    tamper_harden_resnet_cifar100()