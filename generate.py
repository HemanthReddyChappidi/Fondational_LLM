from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

MODEL_PATH = "model"

# -------- DEVICE --------
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon (MPS) detected")
    else:
        device = "cpu"
        print("No GPU found, using CPU")
    return device

device = get_device()
print("Using device:", device)

# -------- LOAD MODEL --------
print("\nLoading model...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)

# Load with appropriate dtype per device
if device == "cuda":
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)  # fp16 on GPU
elif device == "mps":
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)  # MPS needs fp32
else:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)  # CPU fp32

model.to(device)
model.eval()
print("Model loaded successfully!")

# -------- GENERATION --------
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # autocast only supported on CUDA
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                output = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
        else:
            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# -------- CHAT LOOP --------
print("\nYour Mini LLM is ready! Type 'exit' to stop.\n")

while True:
    prompt = input("Prompt: ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    result = generate_text(prompt)
    print("\nModel:", result, "\n")