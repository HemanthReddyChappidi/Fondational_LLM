from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os
import platform

DATA_PATH = "data"
TOKENIZER_PATH = "tokenizer"

os.makedirs(TOKENIZER_PATH, exist_ok=True)

# -------- PLATFORM DETECTION --------
def get_platform_info():
    import torch
    if torch.cuda.is_available():
        platform_name = "CUDA"
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        platform_name = "Apple Silicon (MPS)"
        device_name = "Apple Silicon"
    else:
        platform_name = "CPU"
        device_name = platform.processor() or "Unknown CPU"

    print(f"Platform : {platform_name}")
    print(f"Device   : {device_name}")
    print(f"OS       : {platform.system()} {platform.release()}")
    return platform_name

platform_name = get_platform_info()

# -------- BATCH SIZE BASED ON PLATFORM --------
if platform_name == "CUDA":
    BATCH_SIZE = 5000       # Colab T4 — large RAM, fast I/O
elif platform_name == "Apple Silicon (MPS)":
    BATCH_SIZE = 3000       # Unified memory — moderate batch
else:
    BATCH_SIZE = 1000       # CPU — conservative to avoid RAM issues

print(f"\nUsing batch size: {BATCH_SIZE}")

# -------- LOAD DATASET --------
print("\nDownloading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Dataset loaded! Total samples: {len(dataset):,}")

# -------- BATCH ITERATOR --------
def batch_iterator(batch_size=BATCH_SIZE):
    total = len(dataset)
    for i in range(0, total, batch_size):
        yield dataset[i : i + batch_size]["text"]
        # Progress indicator
        progress = min(i + batch_size, total)
        print(f"  Processing: {progress:,} / {total:,} samples", end="\r")
    print()  # New line after progress

# -------- TRAIN TOKENIZER --------
# Note: Tokenizer training always runs on CPU regardless of platform
# GPU/MPS are not used by the HuggingFace tokenizers library
print("\nTraining tokenizer (runs on CPU — this is normal)...")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# -------- SAVE --------
tokenizer.save_model(TOKENIZER_PATH)
print(f"\nTokenizer saved to: '{TOKENIZER_PATH}'")
print(f"Files created: {os.listdir(TOKENIZER_PATH)}")
print("\nDone! You can now run the training script.")