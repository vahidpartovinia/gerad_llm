import os
import sys
import pickle

import numpy as np
import torch
import math
from tqdm import tqdm

# Paths (adjust as needed)
data_dir = 'data/mmlu'
ckpt_path = 'out-mmlu/ckpt.pt'

# Load metadata
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
max_length = meta['max_length']
num_classes = meta['num_classes']
choice_map = meta['choice_map']

file_size = os.path.getsize(os.path.join(data_dir, 'val_sequences.bin'))
expected_samples = file_size // (max_length * 2)
assert file_size == expected_samples * max_length * 2, "File size does not match expected shape"
print(f"Number of samples: {expected_samples}")

print("max_length:", max_length)
print("val_sequences.bin size:", os.path.getsize(os.path.join(data_dir, 'val_sequences.bin')))

# Load test data (unseen during training)
print("Loading MMLU test split for evaluation...")

# Check if test files exist
test_sequences_path = os.path.join(data_dir, 'test_sequences.bin')
test_labels_path = os.path.join(data_dir, 'test_labels.bin')
test_lengths_path = os.path.join(data_dir, 'test_lengths.bin')

if not os.path.exists(test_sequences_path):
    print("ERROR: test_sequences.bin not found!")
    print("You need to modify your prepare.py to also process the test split.")
    print("The test split should not have been used during training.")
    sys.exit(1)

# Get test file size and calculate samples
test_file_size = os.path.getsize(test_sequences_path)
test_expected_samples = test_file_size // (max_length * 2)
assert test_file_size == test_expected_samples * max_length * 2, "Test file size does not match expected shape"
print(f"Number of test samples: {test_expected_samples}")

test_sequences = np.memmap(
    test_sequences_path,
    dtype=np.uint16,
    mode='r',
    shape=(test_expected_samples, max_length)
)

test_labels = np.fromfile(test_labels_path, dtype=np.uint64)
test_lengths = np.fromfile(test_lengths_path, dtype=np.uint64)
n_test = test_sequences.shape[0]

print(f"Test split loaded: {n_test} samples")

# Load tokenizer
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Load model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_classification import GPTConfig, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Get model's block size
block_size = gptconf.block_size
print(f"Model block size: {block_size}")
print(f"Model config: {gptconf}")

# Debug: Check what the model outputs for a sample
print(f"Model output classes expected: {num_classes}")
print(f"Valid choice map: {choice_map}")

test_file_size = os.path.getsize(test_sequences_path)


def remove_answer_from_sequence(sequence, enc):
    """Remove the answer portion from the sequence"""
    # Decode to find where "Answer:" appears
    text = enc.decode(sequence)
    answer_idx = text.rfind("Answer:")

    if answer_idx != -1:
        # Re-encode just the part before "Answer:"
        text_without_answer = text[:answer_idx].strip()
        return enc.encode(text_without_answer)

    return sequence

# Strategy for handling sequences longer than block_size
def truncate_sequence(sequence, actual_length, block_size, strategy='head'):
    """
    Truncate sequence to fit within block_size

    Args:
        sequence: Full sequence array
        actual_length: Actual length before padding
        block_size: Maximum sequence length model can handle
        strategy: 'tail' (keep end), 'head' (keep start), 'middle' (keep middle)

    Returns:
        Truncated sequence
    """
    # Remove padding first
    sequence = sequence[:actual_length]

    if len(sequence) <= block_size:
        return sequence

    if strategy == 'tail':
        # Keep the last block_size tokens (includes the answer)
        return sequence[-block_size:]
    elif strategy == 'head':
        # Keep the first block_size tokens
        return sequence[:block_size]
    elif strategy == 'middle':
        # Keep some from start and end
        start_tokens = block_size // 3
        end_tokens = block_size - start_tokens
        return np.concatenate([sequence[:start_tokens], sequence[-end_tokens:]])
    else:
        return sequence[-block_size:]  # Default to tail


# Evaluation loop
correct = 0
total_truncated = 0

print(f"\nStarting evaluation on TEST split with truncation strategy: 'head'")
print("=" * 60)

for i in tqdm(range(n_test), desc="Evaluating on test set"):
    actual_length = test_lengths[i]
    sequence = test_sequences[i]

    sequence_without_answer = remove_answer_from_sequence(sequence, enc)

    # Truncate if necessary
    # truncated_sequence = truncate_sequence(sequence, actual_length, block_size, strategy='head')

    # Now truncate the question-only sequence
    truncated_sequence = truncate_sequence(sequence_without_answer,
                                           len(sequence_without_answer),
                                           block_size, strategy='head')

    if len(truncated_sequence) < actual_length:
        total_truncated += 1

    # Convert to tensor and add batch dimension
    input_ids = torch.tensor(truncated_sequence, dtype=torch.long, device=device).unsqueeze(0)
    label = torch.tensor([test_labels[i]], dtype=torch.long, device=device)  # Convert to tensor like in training

    with torch.no_grad():
        logits, _ = model(input_ids, class_targets=label)  # Pass label like in training
        pred = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits[0], dim=-1)

        if confidence[pred] < 0.30:
            print(f"Low confidence (<0.30) at sample {i}: Pred={pred}, Conf={confidence[pred]:.3f}")

    if pred == label.item():
        correct += 1

    # Print first 10 examples and all incorrect predictions
    if i < 10 or pred != label.item():
        truncated_info = f" [TRUNCATED {actual_length}->{len(truncated_sequence)}]" if len(
            truncated_sequence) < actual_length else ""
        print(
            f"Sample {i}: Predicted={pred} ({choice_map[pred]}), Correct={label.item()} ({choice_map[label.item()]}), Conf={confidence[pred]:.3f}{truncated_info}")

        # Show the actual question for first few examples
        if i < 3:
            # Remove padding zeros like in your training code
            # non_zero_tokens = truncated_sequence[truncated_sequence != 0]
            # decoded_text = enc.decode(non_zero_tokens.tolist())
            non_zero_tokens = np.atleast_1d(truncated_sequence[truncated_sequence != 0])
            decoded_text = enc.decode(non_zero_tokens.tolist())
            print(f"  Text preview: {decoded_text[-300:]}...")  # Show last 300 chars
            print()

accuracy = correct / n_test
print("\n" + "=" * 60)
print("EVALUATION RESULTS ON TEST SET")
print("=" * 60)
print(f"Total test samples: {n_test}")
print(f"Samples truncated: {total_truncated} ({total_truncated / n_test * 100:.1f}%)")
print(f"MMLU Test Accuracy: {accuracy:.4f} ({correct}/{n_test})")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

# Additional analysis by confidence
if n_test > 0:
    print(f"\nCorrect predictions: {correct}")
    print(f"Incorrect predictions: {n_test - correct}")

print("=" * 60)