from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

MODEL_PATH = "model"

# -------- LOAD MODEL --------
print("Loading model...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


print("\nYour Mini LLM is ready! Type 'exit' to stop.\n")

while True:
    prompt = input("Prompt: ")

    if prompt.lower() == "exit":
        break

    result = generate_text(prompt)
    print("\nModel:", result, "\n")