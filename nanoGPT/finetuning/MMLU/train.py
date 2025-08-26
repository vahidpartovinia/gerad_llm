import os
import time
import math
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from model import GPTConfig, GPT


train_data = load_dataset('json', data_files='mmlu_train.json')['train']
val_data = load_dataset('json', data_files='mmlu_val.json')['train']



# Load vocabulary
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
meta_vocab_size = meta['vocab_size']

def encode_text(text):
    return [stoi.get(ch, 0) for ch in text]

def encode_label(label):
    return ord(label) - ord('A')

def process(example):
    return {
        'input_ids': encode_text(example['input_text']),
        'label': encode_label(example['label'])
    }

train_data = train_data.map(process)
val_data = val_data.map(process)

block_size = 128
def pad_and_truncate(example):
    ids = example['input_ids']
    if len(ids) < block_size:
        ids += [0] * (block_size - len(ids))
    else:
        ids = ids[:block_size]
    example['input_ids'] = ids
    return example

train_data = train_data.map(pad_and_truncate)
val_data = val_data.map(pad_and_truncate)

import torch

class MMLUDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        label = torch.tensor(item['label'], dtype=torch.long)
        return input_ids, label

train_dataset = MMLUDataset(train_data)
val_dataset = MMLUDataset(val_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

# Specify the checkpoint path for your pretrained model
ckpt_path = '/Users/neginkeshavarz/vsCode/nanoGPT_back/nanoGPT/out/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')
from model import GPTConfig

# These are the ONLY allowed fields in GPTConfig!
allowed_keys = {'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias'}
filtered_config = {k: v for k, v in checkpoint['config'].items() if k in allowed_keys}

gptconf = GPTConfig(**filtered_config)
model = GPT(gptconf)

# ---- PATCH: Partial loading, ignore size mismatch ----
model_dict = model.state_dict()
pretrained_dict = {
    k: v for k, v in checkpoint['model'].items()
    if k in model_dict and v.size() == model_dict[k].size()
}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
# ------------------------------------------------------

out_dir = 'out'
eval_interval = 50
log_interval = 1
eval_iters = 2
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
#dataset = 'shakespeare_char'
gradient_accumulation_steps = 10 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Using device:", device)
dtype = 'float32'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
import os
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    # __file__ is not defined, fallback to current working directory
    script_dir = os.getcwd()
config_path = os.path.join(script_dir, 'configurator.py')

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup

# Always run single device (MPS if available, else CPU)
master_process = True   # Only relevant if you ever want to run multi-process (not needed here)
seed_offset = 0
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

# tf32 only works on CUDA, so skip these on MPS/CPU
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

device_type = 'mps' if 'mps' in device else 'cpu' # for torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# poor man's data loader
#data_dir = os.path.join('data', dataset)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
 
   

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
num_epochs = 3

best_val_acc = 0
print("Starting training loop...")
iter_num = 0
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}...")
    total_loss = 0
    batch_count = 0
    for batch in train_loader:
        if iter_num >= max_iters:
             print(f"Reached {max_iters} iterations, stopping training.")
             break  # breaks out of the inner loop
        model.train()
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _ = model(input_ids, classification=True)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        batch_acc = (preds == labels).float().mean().item()
        iter_num += 1  # Increment the global iteration number
        # Print the global iter_num, batch loss, and accuracy
        print(f"Iter {iter_num} | Batch loss: {loss.item():.4f} | Batch acc: {batch_acc:.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    if batch_count > 0:
        avg_loss = total_loss / batch_count
    else:
        avg_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    scheduler.step()
    # --- Validation ---
    print("====> Starting validation...")
    model.eval()
    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits, _ = model(input_ids, classification=True)
            loss = criterion(logits, labels)  # <-- ADD THIS
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"val_loss sum: {val_loss}, batches: {len(val_loader)}") 
    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)
    print( f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
    print(f"val_loader length: {len(val_loader)}") 

    # --- Saving best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("Best model saved with accuracy:", best_val_acc)


# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0




# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
  
def predict_mcq(model, question, choices, stoi, block_size, device):
    """
    Predict the answer to a multiple-choice question.
    Args:
        model: Your GPT model.
        question: The question string.
        choices: List of answer choices, e.g., ["Charles Dickens", "William Shakespeare", ...]
        stoi: Your vocabulary dictionary.
        block_size: Input length.
        device: 'cpu', 'cuda', or 'mps'
    Returns:
        Predicted letter: 'A', 'B', 'C', or 'D'
    """
    # Build prompt exactly like your training data!
    prompt = question.strip() + "\n"
    for idx, choice in enumerate(choices):
        letter = chr(ord('A') + idx)
        prompt += f"{letter}) {choice}\n"
    # Encode and pad
    input_ids = [stoi.get(ch, 0) for ch in prompt]
    if len(input_ids) < block_size:
        input_ids += [0] * (block_size - len(input_ids))
    else:
        input_ids = input_ids[:block_size]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor, classification=True)
        pred = torch.argmax(logits, dim=1).item()
    label = chr(pred + ord('A'))
    return label, prompt  # return both for inspection

# Example test question:
question = "Who wrote Hamlet?"
choices = [
    "Charles Dickens",
    "William Shakespeare",
    "Mark Twain",
    "Jane Austen"
]
label, formatted_prompt = predict_mcq(model, question, choices, stoi, block_size, device)
print("Prompt used:\n", formatted_prompt)
print("Model prediction:", label)
print("Correct answer: B")



