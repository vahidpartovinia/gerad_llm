import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from model_classification import GPTConfig, GPT


def evaluate_sst2_accuracy(model, tokenizer, dataset, device='cuda', max_samples=None):
    model.eval()
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Try different prompt formats - match your training exactly
        # Option 1: Simple format
        # prompt = example['sentence']

        # Option 2: With explicit labels (if training data included this)
        # prompt = f"{example['sentence']} [SEP] positive" if example['label'] == 1 else f"{example['sentence']} [SEP] negative"

        # Option 3: Your current format
        prompt = f"Review: {example['sentence']}\nSentiment:"

        # Debug: print the first few prompts to compare with training
        if i < 3:
            print(f"Evaluation prompt format: '{prompt}'")

        try:
            input_ids = torch.tensor([tokenizer.encode_ordinary(prompt)], device=device)

            # Debug: print token length
            if i < 3:
                print(f"Token length: {input_ids.shape[1]}")

            # Ensure consistent sequence length with training
            # You might need to pad/truncate to match training block_size
            # Example: if your training uses block_size=1024
            block_size = getattr(gptconf, 'block_size', 1024)  # Check your actual block_size
            if input_ids.shape[1] > block_size:
                input_ids = input_ids[:, :block_size]
            elif input_ids.shape[1] < block_size:
                # Pad with zeros (or whatever padding your training uses)
                pad_length = block_size - input_ids.shape[1]
                padding = torch.zeros((1, pad_length), dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([input_ids, padding], dim=1)
            correct_label = example['label']
            target_tensor = torch.tensor([correct_label], device=device)

            with torch.no_grad():
                # Call model the same way as during training
                logits, _ = model(input_ids, class_targets=target_tensor)
                predicted = torch.argmax(logits, dim=-1).item()

            if predicted == correct_label:
                correct += 1

            total += 1

            if i < 10 or predicted != correct_label:
                confidence = torch.softmax(logits[0], dim=-1)
                print(f"Sample {i}: Text='{example['sentence'][:50]}...'")
                print(f"  Predicted={predicted} (conf: {confidence[predicted]:.3f}), Correct={correct_label}")
                print(f"  Confidence dist: [neg: {confidence[0]:.3f}, pos: {confidence[1]:.3f}]")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, 0


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
val_data = sst2_data["validation"].shuffle(seed=int(time.time()))

# Check class distribution
label_counts = {}
for example in val_data:
    label = example['label']
    label_counts[label] = label_counts.get(label, 0) + 1
print(f"Validation set class distribution: {label_counts}")

# Also check training set
train_data = sst2_data["train"]
train_label_counts = {}
for i, example in enumerate(train_data):
    if i >= 1000:  # Sample first 1000
        break
    label = example['label']
    train_label_counts[label] = train_label_counts.get(label, 0) + 1
print(f"Training set class distribution (first 1000): {train_label_counts}")

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