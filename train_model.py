from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import os

TOKENIZER_PATH = "foundation_llm/tokenizer"
MODEL_PATH = "foundation_llm/model"

print("Loading tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)

tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Building GPT config...")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_ctx=256,
    n_embd=256,
    n_layer=6,
    n_head=8,
)

model = GPT2LMHeadModel(config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-4,
    warmup_steps=100,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Training model...")
trainer.train()

print("Saving model...")
trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("Training complete!")