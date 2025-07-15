import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from datasets import concatenate_datasets

num_proc = 4
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

dataset = load_dataset("cais/mmlu", "all")
#
# test_data = dataset["test"]
#
# split = test_data.train_test_split(test_size=0.1, seed=42)
# val_data = split['test']
# train_data = split['train']

dev_data = dataset['dev']
test_subset = dataset['test'].select(range(2500))

train_data = concatenate_datasets([dev_data, test_subset])
val_data = dataset['validation']

# New
def process(example):
    choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example['choices'])])
    answer_letter = chr(65 + example['answer'])
    text = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer: {answer_letter}"
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    ids.append(enc.eot_token)  # Double separator between samples
    return {'ids': ids, 'len': len(ids)}

tokenized = train_data.map(
    process,
    desc="tokenizing test split",
    num_proc=num_proc,
)

tokenized_val = val_data.map(
    process,
    desc="tokenizing val split",
    num_proc=num_proc,
)

arr_len = np.sum(tokenized['len'], dtype=np.uint64)
filename = os.path.join(os.path.dirname(__file__), 'train.bin')
dtype = np.uint16
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
# total_batches = 1024
total_batches = len(tokenized)

idx = 0
for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
    batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    arr_batch = np.concatenate(batch['ids'])
    arr[idx: idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
arr.flush()

arr_len_val = np.sum(tokenized_val['len'], dtype=np.uint64)
filename_val = os.path.join(os.path.dirname(__file__), 'val.bin')
dtype = np.uint16
arr_val = np.memmap(filename_val, dtype=dtype, mode='w+', shape=(arr_len_val,))
total_batches = 1024

idx = 0
for batch_idx in tqdm(range(total_batches), desc=f'writing {filename_val}'):
    batch = tokenized_val.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    arr_batch = np.concatenate(batch['ids'])
    arr_val[idx: idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
arr_val.flush()
