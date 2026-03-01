from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch
import os

TOKENIZER_PATH = "tokenizer"
MODEL_PATH = "model"
os.makedirs(MODEL_PATH, exist_ok=True)

# -------- DEVICE DETECTION --------
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        print("No GPU found, using CPU")
    print("Using device:", device)
    return device

device = get_device()

# -------- DEVICE-SPECIFIC TRAINING SETTINGS --------
if device == "cuda":
    use_fp16 = True
    use_bf16 = False
    batch_size = 32
    num_workers = 2
elif device == "mps":
    use_fp16 = False
    use_bf16 = False
    batch_size = 16
    num_workers = 0
else:  # CPU
    use_fp16 = False
    use_bf16 = False
    batch_size = 4
    num_workers = 0

print(f"Settings → fp16: {use_fp16} | batch_size: {batch_size} | workers: {num_workers}")

# -------- TOKENIZER --------
print("\nLoading tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token

# -------- DATASET --------
print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Uncomment to use a subset (recommended for CPU or first test runs)
# dataset = dataset.select(range(100_000))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

print("Tokenizing dataset (this may take a few minutes)...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2 if device == "cuda" else 1,
    remove_columns=["text"]
)
tokenized_dataset.set_format(type="torch")

# -------- MODEL --------
print("\nBuilding GPT config...")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_ctx=256,
    n_embd=256,
    n_layer=6,
    n_head=8,
)
model = GPT2LMHeadModel(config)
print(f"Model parameters: {model.num_parameters():,}")

# -------- DATA COLLATOR --------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------- TRAINING ARGS --------
training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-4,
    warmup_steps=200,
    fp16=use_fp16,
    bf16=use_bf16,
    dataloader_num_workers=num_workers,
    prediction_loss_only=True,
    report_to="none",
    lr_scheduler_type="cosine",
    use_cpu=(device == "cpu"),
)

# -------- TRAINER --------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\nTraining model...")
trainer.train()

print("\nSaving model...")
trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print(f"Training complete! Model saved to: {MODEL_PATH}")