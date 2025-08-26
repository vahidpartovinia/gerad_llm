
import os
import pickle
import requests
import numpy as np
from datasets import load_dataset


# Download all MMLU subjects together
from datasets import load_dataset

from datasets import load_dataset

data_files = {
    "train": "/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/data/MMLU/train.csv",
    "validation": "/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/data/MMLU/valid.csv",
    "test": "/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/data/MMLU/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

def preprocess(example):
    choices = [str(choice) if choice is not None else "" for choice in [example['A'], example['B'], example['C'], example['D']]]
    return {
        "input_text": f"prompt: {example['prompt']}\nChoices: {', '.join(choices)}",
        "label": example['answer']
    }

train_data = train_data.map(preprocess)
val_data = val_data.map(preprocess)
test_data = test_data.map(preprocess)

train_data.to_json("/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/mmlu_train.json")
val_data.to_json("/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/mmlu_val.json")
test_data.to_json("/Users/neginkeshavarz/vsCode/gerad_llm/nanoGPT/MMLU/mmlu_test.json")


