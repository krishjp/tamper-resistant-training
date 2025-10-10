from datasets import load_dataset, DatasetDict

def get_imagenet_splits(num_train=80000, num_test=10000, num_eval=10000):
    """
    Loads the full ImageNet dataset from the central HF cache and
    returns specific slices.
    """
    print("Loading full dataset from Hugging Face cache (will download if not present)...")
    
    full_train_dataset = load_dataset("imagenet-1k", split=f"train[:{num_train}]")
    full_test_dataset = load_dataset("imagenet-1k", split=f"test[:{num_test}]")
    full_eval_dataset = load_dataset("imagenet-1k", split=f"validation[:{num_eval}]")

    return DatasetDict({
        "train": full_train_dataset,
        "test": full_test_dataset,
        "validation": full_eval_dataset
    })