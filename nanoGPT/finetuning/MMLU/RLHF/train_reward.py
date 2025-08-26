from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig, RewardModel
import os
from transformers import GPT2TokenizerFast


block_size = 128

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id  # usually 50256
# Load dataset
ds_small = load_dataset("Anthropic/hh-rlhf", split="train")
splits = ds_small.train_test_split(test_size=0.2, seed=42)   # 80/20 split of that 1%
train_ds = splits["train"]
test_ds  = splits["test"]

val_split = train_ds.train_test_split(test_size=0.125, seed=42)  # 12.5% of 80% = 10% of total 1%
train_ds, val_ds = val_split["train"], val_split["test"]





class AnthropicRewardDataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=128):
        self.ds = ds
        self.tok = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ds)
    
    def _encode(self, text):
        return self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False
        )["input_ids"]

    def __getitem__(self, idx):
        item = self.ds[idx]
        chosen_ids = torch.tensor(self._encode(item["chosen"]), dtype=torch.long)
        rejected_ids = torch.tensor(self._encode(item["rejected"]), dtype=torch.long)
        return {"chosen_ids": chosen_ids, "rejected_ids": rejected_ids}
    
def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    chosen = [b["chosen_ids"] for b in batch]
    rejected = [b["rejected_ids"] for b in batch]
    return {
        "chosen_ids": pad_sequence(chosen, batch_first=True, padding_value=pad_id),
        "rejected_ids": pad_sequence(rejected, batch_first=True, padding_value=pad_id),
    }

train_dataset = AnthropicRewardDataset(train_ds, tokenizer, max_length=block_size)
val_dataset   = AnthropicRewardDataset(val_ds,   tokenizer, max_length=block_size)
test_dataset  = AnthropicRewardDataset(test_ds,  tokenizer, max_length=block_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, collate_fn=collate_fn)

gptconf = GPTConfig(
    block_size=block_size,
    vocab_size=50304,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
    bias=False
)
base_model = GPT(gptconf)



ckpt = torch.load("/Users/neginkeshavarz/vsCode/nanoGPT_back/nanoGPT/MMLU/RLHF/ckpt.pt", map_location="cpu")
sd = ckpt.get("model", ckpt)  # handle either flat or nested

reward_model = RewardModel(base_model)   # wrap first
missing, unexpected = reward_model.load_state_dict(sd, strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
reward_model.to(device)


import torch.nn.functional as F
import torch.optim as optim


optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)
num_epochs = 1
def last_nonpad(scores, ids, pad_id):
    if scores.dim() == 3:
        scores = scores.squeeze(-1)
    lengths = (ids != pad_id).sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    return scores.gather(1, lengths.unsqueeze(1)).squeeze(1)

reward_model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        chosen_ids = batch["chosen_ids"].to(device)     # [B, T]
        rejected_ids = batch["rejected_ids"].to(device)  # [B, T]
        
        chosen_reward = last_nonpad(reward_model(chosen_ids), chosen_ids, pad_id=pad_id)
        rejected_reward = last_nonpad(reward_model(rejected_ids), rejected_ids, pad_id=pad_id)
        
        # Pairwise loss: maximize reward difference
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f" Loss: {loss.item():.4f}")

    reward_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            pad_id = tokenizer.pad_token_id  
            cr = last_nonpad(reward_model(chosen_ids), chosen_ids, pad_id=pad_id)
            rr = last_nonpad(reward_model(rejected_ids), rejected_ids, pad_id=pad_id) 
            correct += (cr > rr).sum().item()
            total += chosen_ids.size(0)
    val_acc = correct / total if total else 0.0
    print(f"Epoch {epoch+1}: val accuracy = {val_acc*100:.2f}%")
    reward_model.train()
# ---- Accuracy Evaluation ----
reward_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)

        chosen_reward = last_nonpad(reward_model(chosen_ids), chosen_ids, pad_id=pad_id)
        rejected_reward = last_nonpad(reward_model(rejected_ids), rejected_ids, pad_id=pad_id)

        correct += (chosen_reward > rejected_reward).sum().item()
        total += chosen_ids.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

torch.save(reward_model.state_dict(), "reward_model.pt")
