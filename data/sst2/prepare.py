import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

num_proc = 4
enc = tiktoken.get_encoding("gpt2")

# Load dataset
dataset = load_dataset("glue", "sst2")
train_data = dataset['train']
val_data = dataset['validation']


def process_sst2(example):
    # Format: "Sentence: {text}" - no label in the input
    # The label will be used as target for classification head
    text = f"Sentence: {example['sentence']}"
    ids = enc.encode_ordinary(text)
    return {'ids': ids, 'len': len(ids), 'label': example['label']}


# Process datasets
tokenized = train_data.map(
    process_sst2,
    desc="tokenizing train split",
    num_proc=num_proc,
)

tokenized_val = val_data.map(
    process_sst2,
    desc="tokenizing val split",
    num_proc=num_proc,
)

# Find max length for padding
max_len = max(max(tokenized['len']), max(tokenized_val['len']))
print(f"Max sequence length: {max_len}")


# Convert to padded arrays
def create_padded_data(tokenized_data, filename_prefix):
    n_samples = len(tokenized_data)

    # Create arrays for input sequences and labels
    sequences = np.zeros((n_samples, max_len), dtype=np.uint16)
    labels = np.zeros(n_samples, dtype=np.int64)
    lengths = np.zeros(n_samples, dtype=np.int32)

    for i, example in enumerate(tqdm(tokenized_data, desc=f"Creating {filename_prefix} arrays")):
        seq_len = len(example['ids'])
        sequences[i, :seq_len] = example['ids']
        labels[i] = example['label']
        lengths[i] = seq_len

    # Save arrays
    base_dir = os.path.dirname(__file__)
    sequences.tofile(os.path.join(base_dir, f'{filename_prefix}_sequences.bin'))
    labels.tofile(os.path.join(base_dir, f'{filename_prefix}_labels.bin'))
    lengths.tofile(os.path.join(base_dir, f'{filename_prefix}_lengths.bin'))

    return sequences, labels, lengths


# Create train data
train_sequences, train_labels, train_lengths = create_padded_data(tokenized, 'train')

# Create validation data
val_sequences, val_labels, val_lengths = create_padded_data(tokenized_val, 'val')

# Save metadata
meta = {
    'vocab_size': 50257,  # GPT-2 vocab size
    'max_length': max_len,
    'n_train': len(train_sequences),
    'n_val': len(val_sequences),
    'num_classes': 2
}

import pickle

meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Train dataset length: {len(tokenized)}")
print(f"Val dataset length: {len(tokenized_val)}")
print(f"Max sequence length: {max_len}")
print(f"Vocab size: {meta['vocab_size']}")

# Show example
example = train_data[0]
text = f"Sentence: {example['sentence']}"
print(f"\nExample:")
print(f"Text: {text}")
print(f"Label: {example['label']}")
print(f"Tokens: {enc.encode_ordinary(text)}")