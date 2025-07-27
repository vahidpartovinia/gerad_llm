import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


dataset = load_dataset("Anthropic/hh-rlhf")

example = dataset['train'][0]
print(example.keys())


def extract_anthropic_prompt(prompt_and_response):
    """
    Extract everything up to and including the last 'Assistant:' marker.
    This matches Anthropic SFT/RLHF convention for a prompt.
    """
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    # Include the Assistant: marker in the prompt
    return prompt_and_response[:search_term_idx + len(search_term)]

def extract_anthropic_response(prompt_and_response):
    """
    Extract the text after the last 'Assistant:' marker (the actual response).
    Leading whitespace is stripped.
    """
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    # Everything after the Assistant: marker is the response
    return prompt_and_response[search_term_idx + len(search_term):].lstrip()

def process_anthropic(example):
    """
    Given an example with a 'chosen' field (full Anthropic dialog),
    extract prompt and response, and tokenize for training.
    """
    prompt = extract_anthropic_prompt(example['chosen'])
    response = extract_anthropic_response(example['chosen'])

    # Combine prompt and response (no extra 'Assistant:' needed)
    full_text = f"{prompt}{response}"

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    full_ids = enc.encode_ordinary(full_text)

    # Add end token
    full_ids.append(enc.eot_token)

    return {
        'ids': full_ids,
        'len': len(full_ids)
    }

# for example in dataset['train']:
#     chosen = example['chosen']
#     prompt = extract_anthropic_prompt(chosen)
#     response = extract_anthropic_response(chosen)
#     print("full_text:\n", chosen)
#     print("**********************************************")
#     print("Extracted Prompt:\n", prompt)
#     print("Extracted Response:\n", response)
#     break  # Only print the first example

# Tokenize splits
tokenized_val = [process_anthropic(e) for e in tqdm(dataset['train'], desc="tokenizing val split")]
tokenized_test = [process_anthropic(e) for e in tqdm(dataset['test'], desc="tokenizing test split")]

# Save val.bin
arr_len_val = np.sum([ex['len'] for ex in tokenized_val], dtype=np.uint64)
filename_val = os.path.join(os.path.dirname(__file__), 'val.bin')
dtype = np.uint16
arr_val = np.memmap(filename_val, dtype=dtype, mode='w+', shape=(arr_len_val,))
idx = 0
for ex in tqdm(tokenized_val, desc=f'writing {filename_val}'):
    arr_val[idx: idx + ex['len']] = ex['ids']
    idx += ex['len']
arr_val.flush()

# Save test.bin
arr_len_test = np.sum([ex['len'] for ex in tokenized_test], dtype=np.uint64)
filename_test = os.path.join(os.path.dirname(__file__), 'test.bin')
arr_test = np.memmap(filename_test, dtype=dtype, mode='w+', shape=(arr_len_test,))
idx = 0
for ex in tqdm(tokenized_test, desc=f'writing {filename_test}'):
    arr_test[idx: idx + ex['len']] = ex['ids']
    idx += ex['len']
arr_test.flush()

print(f"Val dataset length: {len(tokenized_val)}")
print(f"Test dataset length: {len(tokenized_test)}")
print(f"Val tokens: {arr_len_val}")
print(f"Test tokens: {arr_len_test}")