import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
#
num_proc = 4
enc = tiktoken.get_encoding("gpt2")

endoftext_token_id = enc.encode_ordinary("<|endoftext|>")[0]
label_token_id = enc.encode_ordinary("Label:")[0]

dataset = load_dataset("glue", "sst2")
train_data = dataset['train']
val_data = dataset['validation']

def process_sst2(example):
    label_str = str(example['label'])
    text = f"Sentence: {example['sentence']}\nLabel: {label_str}<|endoftext|>"
    ids = enc.encode_ordinary(text)
    return {'ids': ids, 'len': len(ids)}

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

# Save train data
arr_len = np.sum(tokenized['len'], dtype=np.uint64)
filename = os.path.join(os.path.dirname(__file__), 'train.bin')
dtype = np.uint16
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

idx = 0
for example in tqdm(tokenized, desc=f'writing {filename}'):
    arr[idx: idx + len(example['ids'])] = example['ids']
    idx += len(example['ids'])
arr.flush()

# Save validation data
arr_len_val = np.sum(tokenized_val['len'], dtype=np.uint64)
filename_val = os.path.join(os.path.dirname(__file__), 'val.bin')
arr_val = np.memmap(filename_val, dtype=dtype, mode='w+', shape=(arr_len_val,))

idx = 0
for example in tqdm(tokenized_val, desc=f'writing {filename_val}'):
    arr_val[idx: idx + len(example['ids'])] = example['ids']
    idx += len(example['ids'])
arr_val.flush()

# Save train labels
train_labels = np.array([ex['label'] for ex in train_data], dtype=np.uint64)
train_labels_path = os.path.join(os.path.dirname(__file__), 'train_labels.bin')
train_labels.tofile(train_labels_path)

# Save validation labels
val_labels = np.array([ex['label'] for ex in val_data], dtype=np.uint64)
val_labels_path = os.path.join(os.path.dirname(__file__), 'val_labels.bin')
val_labels.tofile(val_labels_path)


print(f"Train dataset length: {len(tokenized)}")
print(f"Val dataset length: {len(tokenized_val)}")
print(f"Train tokens: {arr_len}")
print(f"Val tokens: {arr_len_val}")
#
example = train_data[1]
label_str = str(example['label'])
text = f"Sentence: {example['sentence']}\nLabel: {label_str}<|endoftext|>"
print(text)

print('**'*50)