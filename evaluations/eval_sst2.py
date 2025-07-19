import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from model_classification import GPTConfig, GPT

def extract_sentiment(model_output, tokenizer):
    if isinstance(model_output, torch.Tensor):
        text = tokenizer.decode(model_output.tolist())
    else:
        text = model_output

    text_lower = text.lower()
    if "positive" in text_lower or "love" in text_lower or "like" in text_lower:
        return 1
    elif "negative" in text_lower or "not" in text_lower or "hate" in text_lower or "dislike" in text_lower:
        return 0
    return None

def evaluate_sst2_accuracy(model, tokenizer, dataset, device='cuda', max_samples=None):

    model.eval()
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        prompt = f"Review: {example['sentence']}\nSentiment:"
        try:
            input_ids = torch.tensor([tokenizer.encode_ordinary(prompt)], device=device)
            with torch.no_grad():
                logits = model(input_ids)[0]  # shape: (batch, num_classes)
                predicted = torch.argmax(logits, dim=-1).item()
            correct_label = example['label']

            if predicted == correct_label:
                correct += 1

            total += 1

            if i < 10 or predicted != correct_label:
                print(f"Sample {i}: Predicted={predicted}, Correct={correct_label}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, 0  # failed_extractions is always 0 now

# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load checkpoint and model
checkpoint_path = 'out-sst2/ckpt.pt'
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']
print("model_args:", model_args)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load tokenizer
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Load SST-2 test data
print("Loading SST-2 test dataset...")
sst2_data = load_dataset("glue", "sst2")
val_data = sst2_data["validation"]

accuracy, correct, total, failed = evaluate_sst2_accuracy(
    model,
    enc,
    val_data,
    device=device,
    max_samples=100
)

print(f"SST-2 Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Failed extractions: {failed}")

sys.exit(0)