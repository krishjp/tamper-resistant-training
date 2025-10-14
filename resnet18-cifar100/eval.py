import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from datasets import load_dataset
from tqdm.auto import tqdm
import torchvision.transforms as T
import torchvision.models as tv_models

ORIGINAL_MODEL_NAME = "edadaltocg/resnet18_cifar100"
HARDENED_MODEL_PATH = "./resnet-18-cifar100-hardened/final"

# Device selection
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 normalization values
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

# Epsilon in pixel space
EPSILON = 8/255


def get_transform():
    # Keep size at 32x32 for CIFAR-100. If model expects different size, change here.
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])


class HFDatasetTorch(torch.utils.data.Dataset):
    """Wrap a HuggingFace dataset split into a torch Dataset applying transforms on the fly."""
    def __init__(self, hf_dataset, transform=None, label_key=None):
        self.ds = hf_dataset
        self.transform = transform
        # detect label key
        if label_key:
            self.label_key = label_key
        else:
            if 'fine_label' in self.ds.column_names:
                self.label_key = 'fine_label'
            elif 'label' in self.ds.column_names:
                self.label_key = 'label'
            else:
                raise ValueError('No label column found in dataset')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['img']
        # img is a PIL Image or numpy array depending on dataset
        if self.transform:
            img = self.transform(img)
        label = int(ex[self.label_key])
        return {'pixel_values': img, 'labels': label}


def adjust_epsilon_for_normalization(epsilon, std):
    # std: list or tensor of length 3
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return epsilon / std_t


def generate_adversarial_examples(model, images, labels):
    images = images.clone().detach().to(DEVICE)
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs.logits, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data

    eps_scaled = adjust_epsilon_for_normalization(EPSILON, CIFAR100_STD).to(DEVICE)
    perturbed = images + eps_scaled * grad.sign()
    # Clamp in normalized space: (x - mean)/std in [ (0-mean)/std, (1-mean)/std ]
    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1).to(DEVICE)
    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std
    adv_images = torch.max(torch.min(perturbed, max_val), min_val)
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


def main():
    print(f"Using device: {DEVICE}")
    # helper to convert HF checkpoint into a stable torchvision CIFAR resnet state_dict
    def convert_and_save_hf_checkpoint_to_torchvision(model_id_or_path, out_path="converted_resnet18_cifar100.pth"):
        import os
        import torch as _torch
        from huggingface_hub import hf_hub_download

        print(f"Converting checkpoint {model_id_or_path} -> {out_path}")
        # download or locate checkpoint
        if os.path.isdir(model_id_or_path):
            # look for common names
            candidate = None
            for fname in ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.pt']:
                p = os.path.join(model_id_or_path, fname)
                if os.path.exists(p):
                    candidate = p
                    break
            if candidate is None:
                raise FileNotFoundError(f"No checkpoint file found in {model_id_or_path}")
            ckpt_path = candidate
        else:
            ckpt_path = hf_hub_download(repo_id=model_id_or_path, filename='pytorch_model.bin')

        # load checkpoint safely; try safetensors first if available
        sd = None
        try:
            from safetensors.torch import load_file as st_load
            if ckpt_path.endswith('.safetensors'):
                sd = st_load(ckpt_path)
        except Exception:
            pass

        if sd is None:
            # normal torch load; allow the user to run in trusted env
            sd = _torch.load(ckpt_path, map_location='cpu')
        # unwrap if wrapped
        if 'state_dict' in sd:
            sd = sd['state_dict']

        # build torchvision resnet18 CIFAR variant
        tv_resnet = tv_models.resnet18()
        tv_resnet.conv1 = _torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        tv_resnet.maxpool = _torch.nn.Identity()
        tv_resnet.fc = _torch.nn.Linear(tv_resnet.fc.in_features, 100)

        # Decide architecture depending on checkpoint conv1 kernel shape
        ckpt_conv = None
        if 'conv1.weight' in sd:
            w = sd['conv1.weight']
            if hasattr(w, 'shape'):
                ckpt_conv = tuple(w.shape)

        # If checkpoint has 3x3 conv, build CIFAR small-stem; if 7x7, build default resnet and adapt stride/pool
        if ckpt_conv == (64, 3, 3, 3):
            # build CIFAR-style resnet
            tv_resnet = tv_models.resnet18()
            tv_resnet.conv1 = _torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            tv_resnet.maxpool = _torch.nn.Identity()
            tv_resnet.fc = _torch.nn.Linear(tv_resnet.fc.in_features, 100)
            try:
                tv_resnet.load_state_dict(sd, strict=False)
            except Exception as e:
                print('Warning: partial load into CIFAR resnet18:', e)
        else:
            # default resnet (7x7 conv1 expected) - load weights then adapt stride/pool for 32x32 inputs
            tv_resnet = tv_models.resnet18()
            tv_resnet.fc = _torch.nn.Linear(tv_resnet.fc.in_features, 100)
            try:
                tv_resnet.load_state_dict(sd, strict=False)
            except Exception as e:
                print('Warning: partial load into default resnet18:', e)
            # adapt conv1 stride to 1 and remove maxpool so the network works on 32x32
            try:
                old_w = tv_resnet.conv1.weight.data.clone()
                # create new conv with stride=1 but same kernel size
                k = tv_resnet.conv1.kernel_size
                p = tv_resnet.conv1.padding
                tv_resnet.conv1 = _torch.nn.Conv2d(3, 64, kernel_size=k, stride=1, padding=p, bias=False)
                # copy weights (if shapes match)
                if tv_resnet.conv1.weight.data.shape == old_w.shape:
                    tv_resnet.conv1.weight.data.copy_(old_w)
                tv_resnet.maxpool = _torch.nn.Identity()
            except Exception as e:
                print('Failed to adapt conv1 stride/pool for small inputs:', e)

        # save converted weights
        _torch.save(tv_resnet.state_dict(), out_path)
        print(f'Converted checkpoint saved to {out_path}')
        return out_path
    # Load original model: prefer a converted torchvision CIFAR resnet checkpoint per-model
    def load_model_with_fallback(model_id_or_path):
        import os
        import torch as _torch
        print(f"Loading model (preferring converted torchvision): {model_id_or_path}")

        # derive a safe converted filename per model path/id
        base_name = os.path.basename(model_id_or_path.rstrip('/')).replace('.', '_')
        converted_path = os.path.join(os.path.dirname(__file__), f"converted_{base_name}.pth")

        # if converted checkpoint not present, attempt conversion
        if not os.path.exists(converted_path):
            try:
                convert_and_save_hf_checkpoint_to_torchvision(model_id_or_path, out_path=converted_path)
            except Exception as e:
                print('Conversion failed:', e)

        # try loading the converted torchvision checkpoint
        if os.path.exists(converted_path):
            try:
                tv_resnet = tv_models.resnet18()
                tv_resnet.conv1 = _torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                tv_resnet.maxpool = _torch.nn.Identity()
                tv_resnet.fc = _torch.nn.Linear(tv_resnet.fc.in_features, 100)
                sd = _torch.load(converted_path, map_location='cpu')
                tv_resnet.load_state_dict(sd, strict=False)

                class TorchvisionWrapper(_torch.nn.Module):
                    def __init__(self, tv_model):
                        super().__init__()
                        self.tv = tv_model

                    def forward(self, x):
                        logits = self.tv(x)
                        from types import SimpleNamespace
                        return SimpleNamespace(logits=logits)

                model = TorchvisionWrapper(tv_resnet).to(DEVICE)
                print('Loaded converted torchvision model from', converted_path)
                return model
            except Exception as e:
                print('Failed to load converted torchvision model:', e)

        # fallback: try HF AutoModel loader
        try:
            model = AutoModelForImageClassification.from_pretrained(model_id_or_path, ignore_mismatched_sizes=True).to(DEVICE)
            return model
        except Exception as e:
            raise RuntimeError(f'Failed to load model via converted checkpoint and AutoModel: {e}')

    original_model = load_model_with_fallback(ORIGINAL_MODEL_NAME)
    hardened_model = load_model_with_fallback(HARDENED_MODEL_PATH)

    # Small diagnostic: run a few samples through each model and print preds vs labels
    try:
        ds = load_dataset('cifar100')
        sample_ds = ds['test'].select(range(12))
        transform = get_transform()
        imgs = [transform(ex['img']).unsqueeze(0).to(DEVICE) for ex in sample_ds]
        labels = [ex['fine_label'] for ex in sample_ds]
        print('\nQuick diagnostics (first 12 samples):')
        for i, img in enumerate(imgs):
            with torch.no_grad():
                out_orig = original_model(img)
                out_hard = hardened_model(img)
            pred_orig = int(out_orig.logits.argmax(dim=1).cpu().item())
            pred_hard = int(out_hard.logits.argmax(dim=1).cpu().item())
            print(f"idx={i} true={labels[i]:3d} orig_pred={pred_orig:3d} hard_pred={pred_hard:3d}")
    except Exception as e:
        print('Diagnostics failed:', e)

    split_dataset = load_dataset("cifar100")
    val_dataset = split_dataset['test'] # Use the test set for final evaluation

    transform = get_transform()
    val_torch_ds = HFDatasetTorch(val_dataset, transform=transform, label_key='fine_label')
    val_dataloader = torch.utils.data.DataLoader(val_torch_ds, batch_size=64, shuffle=False, num_workers=2)

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