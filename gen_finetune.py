from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

BASE_MODEL = "model"
FINETUNED_MODEL = "finetuned_model"

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

# -------- MODEL SELECTION --------
print("\nChoose model:")
print("1 - Base TinyStories model")
print("2 - Fine-tuned QA model")
choice = input("Enter choice (1 or 2): ")

if choice == "2":
    MODEL_PATH = FINETUNED_MODEL
    print("\nLoading Fine-Tuned QA model...")
else:
    MODEL_PATH = BASE_MODEL
    print("\nLoading Base model...")

# -------- LOAD MODEL --------
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -------- GENERATION --------
def generate_text(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------- CHAT LOOP --------
print("\nChat ready! Type 'exit' to stop.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    prompt = f"### Question: {question} ### Answer:"
    answer = generate_text(prompt)
    answer = answer.replace(prompt, "").strip()
    print("\nModel:", answer, "\n")