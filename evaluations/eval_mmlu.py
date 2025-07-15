import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
import tiktoken
from datasets import load_dataset
from model import GPTConfig, GPT


def extract_answer_choice(model_output, tokenizer):
    """Extract A, B, C, or D from model output"""
    # Convert tokens to text
    if isinstance(model_output, torch.Tensor):
        # Convert tensor to list for tiktoken
        text = tokenizer.decode(model_output.tolist())
    else:
        text = model_output

    # Look for "Answer: X" pattern first
    answer_match = re.search(r'Answer:\s*([ABCD])', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fall back to last occurrence of A, B, C, or D
    choices = re.findall(r'\b([ABCD])\b', text)
    if choices:
        return choices[-1].upper()

    return None


def evaluate_mmlu_accuracy(model, tokenizer, dataset, device='cuda', max_samples=None):
    """Calculate MMLU multiple-choice accuracy"""
    model.eval()
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Create prompt without answer for inference
        choices_str = "\n".join([f"{chr(65 + j)}. {choice}" for j, choice in enumerate(example['choices'])])
        prompt = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer:"

        # Tokenize prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

        # Generate answer using the model's generate method
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.1
            )

        # Extract predicted answer from generated text - convert tensor to list
        generated_text = tokenizer.decode(outputs[0].tolist())
        predicted = extract_answer_choice(generated_text, tokenizer)

        # Convert ground truth answer index to letter
        correct_answer = chr(65 + example['answer'])

        if predicted == correct_answer:
            correct += 1
        total += 1

        if i % 50 == 0:
            print(f"Sample {i}: Predicted={predicted}, Correct={correct_answer}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load checkpoint and model
checkpoint_path = 'out-mmlu/ckpt.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load MMLU test data directly (no preprocessing needed for evaluation)
print("Loading MMLU test dataset...")
mmlu_data = load_dataset("cais/mmlu", "all")  # Use different variable name
test_data = mmlu_data["test"]

print("Running MMLU evaluation...")
accuracy, correct, total = evaluate_mmlu_accuracy(
    model,
    enc,
    test_data,
    device=device,
    max_samples=500
)

print(f"MMLU Accuracy: {accuracy:.4f} ({correct}/{total})")

sys.exit(0)