-----

# Adversarial Hardening of ResNet-18 🛡️

This project demonstrates how to improve the robustness of a ResNet-18 image classification model against adversarial attacks. The technique used is **adversarial training**, where the model is intentionally trained on examples that have been slightly perturbed to fool it. This repository provides scripts to both train a hardened model and evaluate its performance against the original.

The training and evaluation are performed on a subset of the ImageNet dataset and are optimized to run on Intel Arc GPUs (XPU).

-----

## Performance Summary

The primary goal of adversarial hardening is not necessarily to have the highest possible accuracy, but to **reduce the drop in accuracy** when the model is under attack. The results below clearly show that while the original model's performance collapses, the hardened model maintains significantly better performance.

```
==============================
      PERFORMANCE SUMMARY
==============================
Original Model (Clean):       70.15%
Original Model (Attacked):    14.66%  <-- (A 55.49% drop)
------------------------------
Hardened Model (Clean):       54.47%
Hardened Model (Attacked):    29.27%  <-- (Only a 25.20% drop)
==============================
```

### Analysis of Results

  - **Improved Robustness**: The hardened model is **over 2x more accurate** than the original model when subjected to an FGSM attack (29.27% vs 14.66%), proving the training was successful.
  - **Accuracy-Robustness Trade-off**: It's common for adversarially trained models to have slightly lower accuracy on clean, un-attacked data. This is a known trade-off where the model sacrifices some generalization on normal data to gain resilience against attacks.
  - **Potential for Improvement**: As you noted, these results were achieved by training on a small subset of the data (approx. 8,000 adversarial points). The performance of the hardened model could be significantly improved by:
      - Training on a much larger dataset.
      - Retraining on a folded dataset containing both clean and adverse data.
      - Training for more epochs.
      - Tuning hyperparameters like the attack strength (`epsilon`) and learning rate.

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
transformers
datasets
tqdm
accelerate
```

Now, install the dependencies, including the Intel-optimized version of PyTorch.

```bash
# Install PyTorch for Intel GPU
pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/

# Install the rest of the packages
pip install -r requirements.txt
```

-----

## 🚀 Usage

The project is split into two main scripts: one for training and one for evaluation.

**1. Train the Hardened Model**

Run the `tamper_hardening.py` script to begin adversarial training. The script will fine-tune the ResNet-18 model and save the final hardened version to a local directory (e.g., `./resnet-18-imagenet-hardened/final`).

```bash
python tamper_hardening.py
```

**2. Evaluate Model Performance**

After training is complete, run the `eval.py` script. This will load both the original and your newly hardened model, perform clean and adversarial evaluations on both, and print the final performance summary table.

```bash
python eval.py
```

-----

## 📂 Project Structure

```
.
├── tamper_hardening.py    # Main script for adversarial training
├── eval.py                # Script to evaluate and compare models
├── utils/
│   └── loader.py          # Utility for loading ImageNet splits
└── requirements.txt       # Project dependencies
```