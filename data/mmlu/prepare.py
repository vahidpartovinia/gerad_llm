import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from datasets import concatenate_datasets

num_proc = 4
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

dataset = load_dataset("cais/mmlu", "all")

# train on dev, validate on test
train_data = dataset['dev']
val_data = dataset['test']


def process_for_generation(example):
    """Format for generative fine-tuning with proper answer extraction"""
    choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example['choices'])])
    answer_letter = chr(65 + example['answer'])

    # Create input prompt (what model sees during inference)
    prompt = f"Subject: {example['subject']}\nQuestion: {example['question']}\nChoices:\n{choices_str}\nAnswer:"

    # Full sequence for training (prompt + answer)
    full_text = f"{prompt} {answer_letter}"

    # Tokenize
    prompt_ids = enc.encode_ordinary(prompt)
    full_ids = enc.encode_ordinary(full_text)

    # Create labels: -1 for prompt tokens (ignored in loss), actual tokens for answer
    labels = [-1] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Add end token
    full_ids.append(enc.eot_token)
    labels.append(enc.eot_token)

    return {
        'ids': full_ids,
        'labels': labels,
        'len': len(full_ids),
        'prompt_len': len(prompt_ids)
    }


def process_simple(example):
    """Simpler approach - just format the data properly"""
    choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(example['choices'])])
    answer_letter = chr(65 + example['answer'])

    # Better formatting with clear separator
    text = f"Question: {example['question']}\n\nChoices:\n{choices_str}\n\nAnswer: {answer_letter}<|endoftext|>"

    ids = enc.encode_ordinary(text)
    return {'ids': ids, 'len': len(ids)}


tokenized = train_data.map(
    process_simple,
    desc="tokenizing train split",
    num_proc=num_proc,
)

tokenized_val = val_data.map(
    process_simple,
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

print(f"Train dataset length: {len(tokenized)}")
print(f"Val dataset length: {len(tokenized_val)}")
print(f"Train tokens: {arr_len}")
print(f"Val tokens: {arr_len_val}")
