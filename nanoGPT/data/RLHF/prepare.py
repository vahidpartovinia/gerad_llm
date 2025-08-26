from datasets import load_dataset
import json

# Load the hh-rlhf dataset
ds = load_dataset("Anthropic/hh-rlhf")

# Prepare train set
train_samples = []
for ex in ds['train']:
    train_samples.append({
        "chosen": ex["chosen"],
        "rejected": ex["rejected"]
    })

with open("hh_rlhf_train.jsonl", "w") as f:
    for ex in train_samples:
        f.write(json.dumps(ex) + "\n")

# Prepare validation set (from 'test' split)
val_samples = []
for ex in ds['test']:
    val_samples.append({
        "chosen": ex["chosen"],
        "rejected": ex["rejected"]
    })

with open("hh_rlhf_val.jsonl", "w") as f:
    for ex in val_samples:
        f.write(json.dumps(ex) + "\n")