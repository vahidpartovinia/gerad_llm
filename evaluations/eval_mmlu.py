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

    # Fall back to first occurrence of A, B, C, or D after "Answer:"
    answer_pos = text.find("Answer:")
    if answer_pos != -1:
        answer_text = text[answer_pos:]
        choices = re.findall(r'\b([ABCD])\b', answer_text)
        if choices:
            return choices[0].upper()

    # Last resort: any A, B, C, or D in the text
    choices = re.findall(r'\b([ABCD])\b', text)
    if choices:
        return choices[-1].upper()

    return None


def evaluate_mmlu_accuracy(model, tokenizer, dataset, device='cuda', max_samples=None):
    """Calculate MMLU multiple-choice accuracy"""
    model.eval()
    correct = 0
    total = 0
    failed_extractions = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Match the format from your training data preparation
        choices_str = "\n".join([f"{chr(65 + j)}. {choice}" for j, choice in enumerate(example['choices'])])
        prompt = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer:"

        try:
            # Tokenize prompt using tiktoken correctly
            input_ids = torch.tensor([tokenizer.encode_ordinary(prompt)], device=device)

            # Generate answer using the model's generate method
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,  # Increased to capture full answer
                    temperature=0.1
                )

            # Extract only the generated part (not the input prompt)
            generated_tokens = outputs[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(generated_tokens.tolist())

            # Also get the full text for debugging
            full_text = tokenizer.decode(outputs[0].tolist())

            # Extract predicted answer
            predicted = extract_answer_choice(generated_text, tokenizer)

            # Convert ground truth answer index to letter
            correct_answer = chr(65 + example['answer'])

            if predicted == correct_answer:
                correct += 1
            elif predicted is None:
                failed_extractions += 1

            total += 1

            # Debug output for first few samples and failures
            if i < 10 or predicted != correct_answer:
                print(f"Sample {i}: Predicted={predicted}, Correct={correct_answer}")
                print(f"Generated: '{generated_text}'")
                if i < 5:
                    print(f"Full output: '{full_text}'")
                print("---")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            total += 1
            failed_extractions += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, failed_extractions


def evaluate_mmlu_likelihood(model, tokenizer, dataset, device='cuda', max_samples=None):
    """Alternative evaluation using likelihood of each choice"""
    model.eval()
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Create base prompt
        choices_str = "\n".join([f"{chr(65 + j)}. {choice}" for j, choice in enumerate(example['choices'])])
        base_prompt = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer:"

        choice_scores = []

        # Test each choice
        for choice_idx in range(4):
            choice_letter = chr(65 + choice_idx)
            full_prompt = base_prompt + f" {choice_letter}"

            try:
                # Tokenize
                input_ids = torch.tensor([tokenizer.encode_ordinary(full_prompt)], device=device)

                # Get logits
                with torch.no_grad():
                    logits, _ = model(input_ids)

                # Get probability of the choice letter token
                choice_token_id = tokenizer.encode_ordinary(f" {choice_letter}")[0]
                choice_prob = torch.softmax(logits[0, -1, :], dim=-1)[choice_token_id].item()
                choice_scores.append(choice_prob)

            except Exception as e:
                print(f"Error processing choice {choice_letter} for sample {i}: {e}")
                choice_scores.append(0.0)

        # Predict the choice with highest probability
        if choice_scores:
            predicted_idx = torch.argmax(torch.tensor(choice_scores)).item()
            if predicted_idx == example['answer']:
                correct += 1

        total += 1

        if (i + 1) % 100 == 0:
            predicted_letter = chr(65 + predicted_idx) if choice_scores else "?"
            correct_letter = chr(65 + example['answer'])
            print(f"Sample {i}: Predicted={predicted_letter}, Correct={correct_letter}, Scores={choice_scores}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load checkpoint and model
checkpoint_path = 'out-mmlu/ckpt.pt'
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load MMLU test data
print("Loading MMLU test dataset...")
mmlu_data = load_dataset("cais/mmlu", "all")
test_data = mmlu_data["test"]

print("Running MMLU evaluation (generative)...")
accuracy, correct, total, failed = evaluate_mmlu_accuracy(
    model,
    enc,
    test_data,
    device=device,
    max_samples=500
)

print(f"MMLU Accuracy (Generative): {accuracy:.4f} ({correct}/{total})")
print(f"Failed extractions: {failed}")

print("\nRunning MMLU evaluation (likelihood)...")
accuracy_likelihood, correct_likelihood, total_likelihood = evaluate_mmlu_likelihood(
    model,
    enc,
    test_data,
    device=device,
    max_samples=500
)

print(f"MMLU Accuracy (Likelihood): {accuracy_likelihood:.4f} ({correct_likelihood}/{total_likelihood})")

sys.exit(0)