import os
import numpy as np
import tiktoken
import pickle
from tqdm import tqdm
from datasets import load_dataset

# Configuration
num_proc = 4
enc = tiktoken.get_encoding("gpt2")

endoftext_token_id = enc.encode_ordinary("<|endoftext|>")[0]

# Load MMLU dataset
dataset = load_dataset("cais/mmlu", "all")
train_data = dataset['auxiliary_train']
val_data = dataset['validation']
test_data = dataset['test']

# Get all unique subjects for metadata
subjects = list(set(train_data['subject']))
subjects.sort()

# Choice mapping
choice_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def process_mmlu(example):
    """Process MMLU example into the desired format"""
    subject = example['subject']
    question = example['question']
    choices = example['choices']
    answer = example['answer']  # This is already an integer 0-3

    # Format choices as A, B, C, D
    choice_text = ""
    for i, choice in enumerate(choices):
        choice_text += f"{choice_map[i]}) {choice}\n"

    # Create the formatted text
    text = f"Subject: {subject}\nQuestion: {question}\nChoices:\n{choice_text}Answer: {choice_map[answer]}<|endoftext|>"

    ids = enc.encode_ordinary(text)
    return {'ids': ids, 'len': len(ids), 'label': answer}


# Process training data
print("Processing training data...")
tokenized_train = train_data.map(
    process_mmlu,
    desc="tokenizing train split",
    num_proc=num_proc,
)

# Process validation data
print("Processing validation data...")
tokenized_val = val_data.map(
    process_mmlu,
    desc="tokenizing val split",
    num_proc=num_proc,
)

# Create output directory structure
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)

# Determine max length for padding/truncation
max_length = max(max(tokenized_train['len']), max(tokenized_val['len']))
print(f"Maximum sequence length found: {max_length}")


# You might want to set a reasonable max_length limit
# max_length = min(max_length, 512)  # Uncomment to limit max length

def pad_or_truncate(ids, target_length):
    """Pad with zeros or truncate to target length"""
    if len(ids) >= target_length:
        return ids[:target_length]
    else:
        # Pad with zeros
        padded = ids + [0] * (target_length - len(ids))
        return padded


# Save training sequences (fixed length)
print("Saving training sequences...")
train_sequences_path = os.path.join(output_dir, 'train_sequences.bin')
dtype = np.uint16
num_train = len(tokenized_train)
train_arr = np.memmap(train_sequences_path, dtype=dtype, mode='w+', shape=(num_train, max_length))

for i, example in enumerate(tqdm(tokenized_train, desc=f'writing {train_sequences_path}')):
    padded_ids = pad_or_truncate(example['ids'], max_length)
    train_arr[i] = padded_ids
train_arr.flush()

# Save validation sequences (fixed length)
print("Saving validation sequences...")
val_sequences_path = os.path.join(output_dir, 'val_sequences.bin')
num_val = len(tokenized_val)
val_arr = np.memmap(val_sequences_path, dtype=dtype, mode='w+', shape=(num_val, max_length))

for i, example in enumerate(tqdm(tokenized_val, desc=f'writing {val_sequences_path}')):
    padded_ids = pad_or_truncate(example['ids'], max_length)
    val_arr[i] = padded_ids
val_arr.flush()

# Save training labels
print("Saving training labels...")
train_labels = np.array([ex['label'] for ex in tokenized_train], dtype=np.uint64)
train_labels_path = os.path.join(output_dir, 'train_labels.bin')
train_labels.tofile(train_labels_path)

# Save validation labels
print("Saving validation labels...")
val_labels = np.array([ex['label'] for ex in tokenized_val], dtype=np.uint64)
val_labels_path = os.path.join(output_dir, 'val_labels.bin')
val_labels.tofile(val_labels_path)

# Save training lengths
print("Saving training lengths...")
train_lengths = np.array(tokenized_train['len'], dtype=np.uint64)
train_lengths_path = os.path.join(output_dir, 'train_lengths.bin')
train_lengths.tofile(train_lengths_path)

# Save validation lengths
print("Saving validation lengths...")
val_lengths = np.array(tokenized_val['len'], dtype=np.uint64)
val_lengths_path = os.path.join(output_dir, 'val_lengths.bin')
val_lengths.tofile(val_lengths_path)

max_len = max(max(tokenized_train['len']), max(tokenized_val['len']))

# Create and save metadata
print("Saving metadata...")
meta = {
    'vocab_size': 50257,  # GPT-2 vocab size
    'max_length': max_length,
    'n_train': num_train,
    'n_val': num_val,
    'num_classes': 4,  # A, B, C, D choices
    'subjects': subjects,
    'num_subjects': len(subjects),
    'choice_map': choice_map,
    'endoftext_token': endoftext_token_id,
}

meta_path = os.path.join(output_dir, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

# Print summary statistics
print("\n" + "=" * 50)
print("MMLU Dataset Processing Complete!")
print("=" * 50)
print(f"Training examples: {len(tokenized_train)}")
print(f"Validation examples: {len(tokenized_val)}")
print(f"Maximum sequence length: {max_length}")
print(f"Number of subjects: {len(subjects)}")
print(f"Vocabulary size: {enc.n_vocab}")
print("\nFiles created:")
print(f"  - {train_sequences_path}")
print(f"  - {val_sequences_path}")
print(f"  - {train_labels_path}")
print(f"  - {val_labels_path}")
print(f"  - {train_lengths_path}")
print(f"  - {val_lengths_path}")
print(f"  - {meta_path}")

# Show example of processed data
print("\n" + "=" * 50)
print("Example processed training sample:")
print("=" * 50)
example = tokenized_train[0]
decoded_text = enc.decode(example['ids'])
print(decoded_text)
print(f"\nLabel: {example['label']} (Answer: {choice_map[example['label']]})")
print(f"Sequence length: {example['len']} tokens")
print("=" * 50)