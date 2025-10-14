# Adversarial Hardening of ResNet-18

This project demonstrates how to improve the robustness of a ResNet-18 image classification model against adversarial attacks. The technique used is **adversarial training**, where the model is intentionally trained on examples that have been slightly perturbed to fool it. This repository provides scripts to both train a hardened model and evaluate its performance against the original.

The training and evaluation are performed on two different datasets: a **subset of ImageNet** and **CIFAR-100**.

## Performance Summary

The primary goal of adversarial hardening is to **reduce the drop in accuracy** when the model is under attack. The results below show a consistent pattern across both datasets: while the original model's performance collapses under an adversarial attack, the hardened model maintains significantly better performance.

### ImageNet (Subset)

```
==============================
  PERFORMANCE SUMMARY (IMAGENET)
==============================
Original Model (Clean):       70.15%
Original Model (Attacked):    14.66%  <-- (A 55.49% drop)
------------------------------
Hardened Model (Clean):       54.47%
Hardened Model (Attacked):    29.27%  <-- (Only a 25.20% drop)
==============================
```

### CIFAR-100

```
==============================
  PERFORMANCE SUMMARY (CIFAR-100)
==============================
Original Model (Clean):       69.79%
Original Model (Attacked):     9.73%  <-- (60.06% drop)
------------------------------
Hardened Model (Clean):       66.41%
Hardened Model (Attacked):    15.25%  <-- (51.16% drop)
==============================
```

### CIFAR-100 (Top 5 accuracy)

```
==============================
 PERFORMANCE SUMMARY (CIFAR-100)
==============================
Original Model (Clean):      88.79%
Original Model (Attacked):   25.58%  <-- (63.21% drop)
------------------------------
Hardened Model (Clean):      86.84%
Hardened Model (Attacked):   37.64%  <-- (49.20% drop)
==============================
```
-----

### Analysis of Results

  * **Improved Robustness**: In both experiments, the hardened model is significantly more accurate than the original when attacked. On the ImageNet subset, it's **over 2x more accurate** (29.27% vs 14.66%), and on CIFAR-100, it's **over 1.5x more accurate** (15.25% vs 9.73%). This proves the training was successful.
  * **Accuracy-Robustness Trade-off**: A common trade-off is observed in both cases, where the hardened models have slightly lower accuracy on clean, un-attacked data. The models sacrifice some generalization on normal data to gain resilience against attacks.
  * **Potential for Improvement**: The performance of the hardened models could be further improved by training for more epochs, using more advanced attack methods for training (e.g., PGD), and tuning hyperparameters like the attack strength (`epsilon`) and learning rate.

-----

## 🛠️ Setup and Installation

Follow these steps to set up the environment and run the project.

**1. Clone the Repository**

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

**2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**

First, create a `requirements.txt` file with the following contents:

**`requirements.txt`**

```
torch
torchvision
torchaudio
timm
datasets
tqdm
safetensors
```

Now, install the dependencies.

```bash
pip install -r requirements.txt
```

-----

## Usage

The project is split into directories for each dataset. Choose the one you wish to work with.

**1. Train a Hardened Model**

Run the appropriate training script to begin adversarial fine-tuning. This will save the final hardened version to a local directory.

*For CIFAR-100:*

```bash
python resnet18-cifar100/tamper_harden_resnet_cifar100.py
```

*For ImageNet:*

```bash
python resnet18-imagenet/tamper_hardening.py
```

**2. Evaluate Model Performance**

After training, run the corresponding evaluation script. It will load the original and hardened models, perform clean and adversarial evaluations, and print the final performance table.

*For CIFAR-100:*

```bash
python resnet18-cifar100/eval.py
```

*For ImageNet:*

```bash
python resnet18-imagenet/eval.py
```

-----

## Project Structure

```
.
├── resnet18-cifar100/
│   ├── tamper_harden_resnet_cifar100.py  # Training script for CIFAR-100
│   └── eval.py                           # Evaluation script for CIFAR-100
├── resnet18-imagenet/
│   ├── tamper_hardening.py               # Training script for ImageNet
│   └── eval.py                           # Evaluation script for ImageNet
└── requirements.txt                      # Project dependencies
```