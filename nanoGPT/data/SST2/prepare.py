import os
import pickle
import numpy as np
from datasets import load_dataset

# Load SST-2
dataset = load_dataset("glue", "sst2")

train_sentences = dataset['train']['sentence']
train_labels = dataset['train']['label']
val_sentences = dataset['validation']['sentence']
val_labels = dataset['validation']['label']

# Build character vocab from training sentences
all_text = ''.join(train_sentences)
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print("All unique characters:", ''.join(chars))
print("Vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi.get(c, 0) for c in s]  # unknown chars as 0

def decode(l):
    return ''.join([itos.get(i, '') for i in l])

max_length = 256  # adjust as needed

def pad(seq, max_len=max_length):
    return seq[:max_len] + [0]*(max_len - len(seq))

# Encode and pad
train_ids = np.array([pad(encode(s)) for s in train_sentences], dtype=np.uint16)
val_ids = np.array([pad(encode(s)) for s in val_sentences], dtype=np.uint16)
train_labels = np.array(train_labels, dtype=np.uint8)
val_labels = np.array(val_labels, dtype=np.uint8)

# Save to bin files
save_dir = os.path.join('data', 'SST2')  # or your dataset name
os.makedirs(save_dir, exist_ok=True)
train_ids.tofile(os.path.join(save_dir, 'train.bin'))
train_labels.tofile(os.path.join(save_dir, 'train_labels.bin'))
val_ids.tofile(os.path.join(save_dir, 'val.bin'))
val_labels.tofile(os.path.join(save_dir, 'val_labels.bin'))

# Save meta info
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'max_length': max_length
}
with open(os.path.join(save_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done preparing SST-2 data for character-level modeling.")