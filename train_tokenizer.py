from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

DATA_PATH = "data"
TOKENIZER_PATH = "tokenizer"

os.makedirs(TOKENIZER_PATH, exist_ok=True)

print("Downloading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

print("Training tokenizer...")

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

tokenizer.save_model(TOKENIZER_PATH)

print("Tokenizer saved!")