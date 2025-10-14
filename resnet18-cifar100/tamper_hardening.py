import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import timm


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
EPSILON = 8/255


def get_train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])


def get_eval_transform():
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])


class HFDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, label_key='fine_label'):
        self.ds = hf_dataset
        self.transform = transform
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['img']
        if self.transform:
            img = self.transform(img)
        label = int(ex[self.label_key])
        return img, label


def epsilon_normalized(epsilon):
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return epsilon / std


def tamper_harden_resnet_cifar100(output_dir='./resnet-18-cifar100-hardened', epochs=1, batch_size=128, lr=0.1):
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'
    print('Using device', device)

    print("Loading pre-trained ResNet-18 for CIFAR-100 from timm...")
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    
    model = model.to(device)

    # dataset
    ds = load_dataset('cifar100')
    train_ds = HFDatasetTorch(ds['train'], transform=get_train_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs), int(0.75*epochs)], gamma=0.1)

    eps_norm = epsilon_normalized(EPSILON).to(device)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs.requires_grad = True
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            # compute grad of loss wrt inputs without calling backward (avoid double-backward)
            # retain_graph=True so we can still backpropagate loss_final later
            grad = torch.autograd.grad(loss, imgs, retain_graph=True, create_graph=False)[0]
            perturbed = imgs + eps_norm * grad.sign()
            # clamp in normalized space
            mean = torch.tensor(CIFAR100_MEAN).view(1,3,1,1).to(device)
            std = torch.tensor(CIFAR100_STD).view(1,3,1,1).to(device)
            min_val = (0.0 - mean) / std
            max_val = (1.0 - mean) / std
            adv = torch.max(torch.min(perturbed, max_val), min_val).detach()

            logits_adv = model(adv)
            loss_adv = F.cross_entropy(logits_adv, labels)

            loss_final = 0.5 * (loss + loss_adv)
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f"{loss_final.item():.4f}"})

        scheduler.step()

    # save model state dict in hardened folder
    os.makedirs(os.path.join(output_dir, 'final'), exist_ok=True)
    out_path = os.path.join(output_dir, 'final', 'model.safetensors')
    try:
        from safetensors.torch import save_file as st_save
        st_save(model.state_dict(), out_path)
        print('Saved safetensors model to', out_path)
    except Exception:
        torch.save(model.state_dict(), os.path.join(output_dir, 'final', 'pytorch_model.bin'))
        print('Saved torch state_dict to', os.path.join(output_dir, 'final', 'pytorch_model.bin'))
    # also save a converted checkpoint that eval.py prefers
    converted_out = os.path.join(os.path.dirname(__file__), 'converted_final.pth')
    torch.save(model.state_dict(), converted_out)
    print('Also saved converted checkpoint to', converted_out)


if __name__ == '__main__':
    tamper_harden_resnet_cifar100(epochs=5, batch_size=128, lr=1e-4)