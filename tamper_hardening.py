from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from utils.loader import get_imagenet_splits
import torch
import torch.optim as optim

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18", use_fast=True)

class AdversarialTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs['pixel_values'].clone().detach()
        labels = inputs['labels'].clone().detach()
        
        pixel_values.requires_grad = True
        outputs_clean = model(pixel_values)
        loss_clean = torch.nn.functional.cross_entropy(outputs_clean.logits, labels)
        
        grad = torch.autograd.grad(loss_clean, pixel_values,
                                   retain_graph=True, create_graph=False)[0]
        # FGSM
        epsilon_train = torch.tensor(8/255, device=pixel_values.device, dtype=pixel_values.dtype)
        perturbation = epsilon_train * torch.sign(grad.data)
        adv_pixel_values = torch.clamp(pixel_values.detach() + perturbation, min=0, max=1)

        outputs_adv = model(adv_pixel_values)
        loss_adv = torch.nn.functional.cross_entropy(outputs_adv.logits, labels)
        
        loss = (loss_clean + loss_adv) / 2

        return (loss, outputs_adv) if return_outputs else loss

def transform(examples):
        examples["pixel_values"] = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")['pixel_values']
        return examples

def tamper_harden_resnet():
    model_name = "microsoft/resnet-18"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model = model.to(torch.float32)

    split_dataset = get_imagenet_splits()
    train_dataset = split_dataset['train']

    processed_train_dataset = train_dataset.map(transform, batched=True)
    is_xpu_available = torch.xpu.is_available()
    if not is_xpu_available:
        print("XPU is not available. Exiting...")
        return
    print(f"Device Name: {torch.xpu.get_device_name(0)}")


    base_path = "./resnet-18-imagenet-hardened"
    training_args = TrainingArguments(
        output_dir=base_path,
        num_train_epochs=3,
        per_device_train_batch_size=32 if is_xpu_available else 16,
        logging_steps=100,
        save_strategy="epoch",
        learning_rate=2e-5,
        label_names=["labels"],
        fp16=False,
        bf16=is_xpu_available,
    )
    optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate, fused=False)
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        optimizers=(optimizer, None)
    )
    trainer.train()

    hardened_model_path = base_path + "/final"
    trainer.save_model(hardened_model_path)
    print(f"Hardened model saved to {hardened_model_path}")

if __name__ == "__main__":
    tamper_harden_resnet()