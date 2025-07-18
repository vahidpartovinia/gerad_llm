import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

import re

def parse_sst2_line(line):
    label = int(line[1])
    sentence = ' '.join(re.findall(r'\b[A-Za-z0-9\-\'\.\,]+\b', line))
    return sentence, label


# Load Shakespeare text
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/input.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

# Load SST2 and extract raw sentences
sst2_sentences = []
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            s, _ = parse_sst2_line(line)
            sst2_sentences.append(s)

from datasets import load_dataset

# Load Anthropic helpful-harmless dataset
hh = load_dataset("Anthropic/hh-rlhf", split="train")
print("Loading HH dataset...")



chars = sorted(list(set(shakespeare_text)))
chars = ['<pad>', '<unk>'] + chars  # add pad and unknown tokens at start

vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]  # use <unk> for unseen chars
decode = lambda l: ''.join([itos[i] for i in l if i > 1])   # skip pad/unk for string output

# Train and test splits
data = torch.tensor(encode( shakespeare_text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, n_embd):
        super().__init__()
        self.token_embedding_table = pretrained_model.token_embedding_table
        self.position_embedding_table = pretrained_model.position_embedding_table
        self.blocks = pretrained_model.blocks
        self.ln_f = pretrained_model.ln_f
        self.classifier = nn.Linear(n_embd, 1)  # Single logit for binary sentiment
    def forward(self, idx):
        T = idx.shape[1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # [B, n_embd], mean-pooling over sequence
        logit = self.classifier(x).squeeze(-1)  # [B]
        return logit   

class RewardModel(nn.Module):
    def __init__(self, base_model, n_embd):
        super().__init__()
        self.base = SentimentClassifier(base_model, n_embd)

    def forward(self, idx):
        return self.base(idx)  # Returns score directly
        

pretrain_path = 'shakespeare_pretrained.pt'
model = BigramLanguageModel()
m = model.to(device)

if os.path.exists(pretrain_path):
    print("Loading pretrained model weights...")
   
else:
    print("No pretrained model found. Starting pretraining...")
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), pretrain_path)
    print("Pretraining finished and model saved.")



import re

def parse_sst2_line(line):
    label = int(line[1])
    sentence = ' '.join(re.findall(r'\b[A-Za-z0-9\-\'\.\,]+\b', line))
    return sentence, label

sst2_sentences, sst2_labels = [], []
with open('/Users/neginkeshavarz/vsCode/RA/NanoGPT/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            s, l = parse_sst2_line(line)
            sst2_sentences.append(s)
            sst2_labels.append(1 if l >= 3 else 0)  # binary label

def encode_and_pad(sentence, block_size):
    ids = encode(sentence)
    if len(ids) < block_size:
        ids += [0] * (block_size - len(ids))  # Pad with zeros
    else:
        ids = ids[:block_size]
    return ids

X = [encode_and_pad(sent, block_size) for sent in sst2_sentences]
Y = sst2_labels
X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

import numpy as np

# Shuffle and split indices
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

split = int(0.9 * len(X))  # 90% train, 10% val

train_idx, val_idx = indices[:split], indices[split:]
X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
X_val, Y_val = X_tensor[val_idx], Y_tensor[val_idx]

from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Use the already loaded or pretrained model!
classifier_model = SentimentClassifier(model, n_embd).to(device)

optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()  # For binary sentiment

num_epochs = 3  # or as many as you want



for epoch in range(num_epochs):
    # === TRAINING ===
    classifier_model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = classifier_model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    train_loss = total_loss / total
    train_acc = correct / total

    # === VALIDATION ===
    classifier_model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier_model(xb)
            loss = loss_fn(logits, yb)
            val_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, train acc={train_acc:.4f} | val loss={val_loss:.4f}, val acc={val_acc:.4f}")

    classifier_model.train()  # back to train mode



torch.save(classifier_model.state_dict(), 'finetuned_on_sst2.pt')

reward_model = RewardModel(model, n_embd).to(device)
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# Convert HH data to encoded tensors
def encode_reward_pair(chosen, rejected):
    chosen_ids = encode_and_pad(chosen, block_size)
    rejected_ids = encode_and_pad(rejected, block_size)
    return torch.tensor(chosen_ids), torch.tensor(rejected_ids)

pairs = []
for item in hh.select(range(10000)):  # Limit for speed; increase for full
    chosen, rejected = item['chosen'], item['rejected']
    ch, rj = encode_reward_pair(chosen, rejected)
    pairs.append((ch, rj))

# Training loop
reward_model.train()
for epoch in range(1):
    total_loss = 0
    for chosen_ids, rejected_ids in pairs:
        chosen_ids = chosen_ids.unsqueeze(0).to(device)
        rejected_ids = rejected_ids.unsqueeze(0).to(device)

        chosen_score = reward_model(chosen_ids)
        rejected_score = reward_model(rejected_ids)
        
        # Preferred score should be higher
        loss_fn = nn.MarginRankingLoss(margin=1.0)
        target = torch.ones_like(chosen_score)
        loss = loss_fn(chosen_score, rejected_score, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Reward model epoch complete. Avg loss: {total_loss / len(pairs):.4f}")

# === PPO STEP PREPARATION ===

def generate_response(prompt, max_new_tokens=64):
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens)
    return decode(out[0].tolist())

def compute_reward(response):
    input_ids = torch.tensor([encode_and_pad(response, block_size)], dtype=torch.long).to(device)
    with torch.no_grad():
        reward = reward_model(input_ids)
    return reward.item()
def ppo_loss(old_log_probs, new_log_probs, advantage, epsilon=0.2):
    ratio = (new_log_probs - old_log_probs).exp()
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return -torch.min(unclipped, clipped).mean()

# PPO Training loop
ppo_steps = 5000
baseline = 0.0  # Running average of rewards
clip_epsilon = 0.2
ppo_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for step in range(ppo_steps):
    full_text = hh[step]['chosen']
    prompt = full_text.split('Assistant:')[0].strip()
    response = generate_response(prompt)
    reward = compute_reward(response)
    
    baseline = 0.9 * baseline + 0.1 * reward  # EMA(Exponential Moving Average) baseline
    advantage = reward - baseline

    input_ids = torch.tensor([encode_and_pad(response, block_size)], dtype=torch.long).to(device)
    
    # Get old logits before update
    with torch.no_grad():
        old_logits, _ = model(input_ids)
        old_log_probs = F.log_softmax(old_logits, dim=-1)

    # Forward pass again for new logits
    new_logits, _ = model(input_ids)
    
    new_log_probs = F.log_softmax(new_logits, dim=-1)

    # For simplicity, use average log-probs across all tokens
    loss = ppo_loss(old_log_probs.mean(), new_log_probs.mean(), torch.tensor(advantage).to(device), epsilon=clip_epsilon)

    ppo_optimizer.zero_grad()
    loss.backward()
    ppo_optimizer.step()

    if step % 50 == 0:
        print(f"[PPO Step {step}] Reward: {reward:.4f}, Baseline: {baseline:.4f}, Loss: {loss.item():.4f}")

        

    # generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))