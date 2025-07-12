# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
import os

import numpy as np
from transformers.integrations import tiktoken
import tiktoken
from tqdm import tqdm

num_proc = 4

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all")

    test_data = dataset["test"]

    split = test_data.train_test_split(test_size=0.1, seed=42)
    val_data = split['test']
    train_data = split['train']

    def process(example):
        # Combine question, choices, and answer into a single string
        choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example['choices'])])
        text = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer: {example['answer']}"
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
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
    total_batches = 1024

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

