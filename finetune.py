from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

MODEL_PATH = "model"
FINETUNED_PATH = "finetuned_model"
DATA_PATH = "finetune_data/qa_dataset.jsonl"

# -------- DEVICE DETECTION --------
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        print("No GPU found, using CPU")
    print("Using device:", device)
    return device

device = get_device()

# -------- DEVICE-SPECIFIC SETTINGS --------
if device == "cuda":
    use_fp16 = True
    use_bf16 = False
    batch_size = 16
    num_workers = 2
elif device == "mps":
    use_fp16 = False
    use_bf16 = False
    batch_size = 4
    num_workers = 0
else:  # CPU
    use_fp16 = False
    use_bf16 = False
    batch_size = 2
    num_workers = 0

print(f"Settings → fp16: {use_fp16} | batch_size: {batch_size} | workers: {num_workers}")

# -------- MODEL & TOKENIZER --------
print("Loading base model...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)

# -------- DATASET --------
print("Loading fine-tuning dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=["text"],
    num_proc=1
)
tokenized_dataset.set_format(type="torch")

# -------- DATA COLLATOR --------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------- TRAINING ARGS --------
training_args = TrainingArguments(
    output_dir=FINETUNED_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_total_limit=2,
    fp16=use_fp16,
    bf16=use_bf16,
    dataloader_num_workers=num_workers,
    report_to="none",
    no_cuda=(device != "cuda"),
    use_mps_device=(device == "mps"),
)

# -------- TRAINER --------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Fine-tuning model...")
trainer.train()

print("Saving fine-tuned model...")
trainer.save_model(FINETUNED_PATH)
tokenizer.save_pretrained(FINETUNED_PATH)
print("Fine-tuning complete!")