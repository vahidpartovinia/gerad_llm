from model import GPT, GPTConfig, RewardModel
import copy
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
import random

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)

def top_k_logits(logits, k=50):
    if k <= 0 or k >= logits.size(-1):
        return logits
    vals, idx = torch.topk(logits, k)
    cutoff = vals[:, -1].unsqueeze(-1)
    return torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)

def logprobs_from_logits(logits, tokens):
    # logits: [B, T, V], tokens: [B, T]
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)

def sequence_mask_from_lengths(lengths, max_len):
    # lengths are prompt lengths (incl. padding mask); we mask only generated tokens
    B = lengths.size(0)
    mask = torch.zeros((B, max_len-1), dtype=torch.float32, device=lengths.device)
    for i in range(B):
        L = max(int(lengths[i].item()), 1)
        mask[i, L-1:] = 1.0                 # generated tokens begin at index L-1 in tokens_out
    return mask


def ppo_clip_loss(old_logprobs, new_logprobs, advantages, mask, eps_clip=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    return -masked_mean(torch.min(surr1, surr2), mask)

def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)
    return -(probs * logprobs).sum(dim=-1)  # [B,T]
# device
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)


def get_logprobs(logits, actions):
    # logits: [B, T, vocab]
    # actions: [B, T]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


vocab_size = 50304  
# === Load GPT model ===
gptconf = GPTConfig(
    block_size=128,
    vocab_size=vocab_size,
    n_layer=4,        # match your reward training (or whatever was inferred)
    n_head=4,         # match reward training if that's what you used
    n_embd=256,
    dropout=0.1,
    bias=False
)

# Policy (trainable)
policy = GPT(gptconf).to(device)
# simple value head that reads the last-layer hidden states (size n_embd)
policy.v_head = nn.Linear(gptconf.n_embd, 1, bias=False).to(device)

# Reference policy (frozen) – start as a copy of the initial policy
ref_policy = copy.deepcopy(policy).to(device)
for p in ref_policy.parameters():
    p.requires_grad = False
ref_policy.eval()

# Reward model (frozen) – built on its own backbone
rm_backbone = GPT(gptconf).to(device)
reward_model = RewardModel(rm_backbone).to(device)
reward_model.load_state_dict(torch.load("reward_model.pt", map_location=device))
for p in reward_model.parameters():
    p.requires_grad = False
reward_model.eval()



tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id   




# ✅ Now you can build the PPO training loop here
import copy

ds = load_dataset("Anthropic/hh-rlhf", split="train[:1%]")
class AnthropicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item.get("prompt", item.get("question", ""))  # fall back to 'question'
        # encode prompt
        input_ids = self.tokenizer.encode(prompt)[:self.max_length]
        if len(input_ids) == 0:
           input_ids = [self.tokenizer.pad_token_id]
        return {"input_ids": input_ids}
train_dataset = AnthropicDataset(ds, tokenizer)
def collate_fn(batch):
    # batch is a list of dicts like {"input_ids": [ids]}
    return tokenizer.pad(batch, padding=True, return_tensors='pt')

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

@torch.no_grad()
def generate_with_policy(policy, input_ids, max_new_tokens=64,
                         temperature=0.7, top_k=50, top_p=0.9,
                         repetition_penalty=1.1, eos_id=None):
    policy.eval()
    x = input_ids.to(device)
    eos_id = eos_id if eos_id is not None else tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        x_cond = x[:, -gptconf.block_size:]
        logits = policy(x_cond)[0][:, -1, :]

        # repetition penalty
        for b in range(x.size(0)):
            logits[b, x[b]] /= repetition_penalty

        logits = logits / max(temperature, 1e-8)

        # top-k filtering
        if 0 < top_k < logits.size(-1):
            vals, _ = torch.topk(logits, top_k)
            cutoff = vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)

        # top-p (nucleus sampling)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[:, 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        x = torch.cat([x, next_tok], dim=1)

        if eos_id is not None and (next_tok == eos_id).all():
            break

    return x

import torch
from torch.utils.data import DataLoader


optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
grad_clip     = 1.0
ppo_epochs    = 4
eps_clip      = 0.2
entropy_coef  = 0.01     # can decay later to 0.001
kl_coef       = 0.02     # will be adapted
target_kl     = 0.02
max_new_tokens = 32      # shorter rollouts => lower variance
mini_batch_size = 8      # PPO mini-batches inside each batch
num_epochs    = 3
gamma         = 0.99

all_losses = []



# === Evaluation helpers ===
@torch.no_grad()
def evaluate_model(policy, tokenizer, reward_model, prompts, max_new_tokens=32):
    policy.eval()
    reward_model.eval()

    scores = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = generate_with_policy(policy, enc.input_ids, max_new_tokens=max_new_tokens)

        # Slice off the prompt
        prompt_len = enc.input_ids.size(1)
        completion_ids = out[0, prompt_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        score = reward_model(out).mean().item()
        scores.append(score)

        print("\n" + "-"*60)
        print(f"Prompt:\n{prompt}")
        print("\nCompletion:\n" + completion)
        print(f"\nReward score: {score:+.3f}")

    avg = sum(scores)/max(len(scores), 1)
    print("\n" + "="*60)
    print(f"Average reward over {len(prompts)} prompts: {avg:+.3f}")
    print("="*60 + "\n")
    return avg

def evaluate_after_epoch(epoch, policy, tokenizer, reward_model):
    eval_prompts = [
        "Human: Explain reinforcement learning in simple terms.\nAssistant:",
        "Human: Write a haiku about the ocean.\nAssistant:",
        "Human: List three benefits of regular exercise.\nAssistant:",
        "Human: Describe the importance of unit testing in code.\nAssistant:",
        "Human: Give a polite response if you can’t help with a request.\nAssistant:"
    ]
    print(f"\n[Eval] Epoch {epoch} — generating samples and reward scores...")
    avg = evaluate_model(policy, tokenizer, reward_model, eval_prompts, max_new_tokens=32)
    print(f"[Eval] Epoch {epoch} — average reward: {avg:+.3f}\n")




policy.train()
global_step = 0

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)        # [B, T_prompt]
        attention_mask = batch["attention_mask"].to(device)
        B = input_ids.size(0)

        # --- Generate completions ---
        with torch.no_grad():
            sequences = generate_with_policy(policy, input_ids, max_new_tokens=max_new_tokens, top_k=50, temperature=1.0)
            # Prepare teacher-forcing pairs
            tokens_in  = sequences[:, :-1]               # [B, T-1]
            tokens_out = sequences[:,  1:]               # [B, T-1]

            # Which positions are generated (exclude prompt)
            prompt_lens = attention_mask.sum(dim=1)      # [B]
            gen_mask = sequence_mask_from_lengths(prompt_lens, sequences.size(1))  # [B, T-1]

            # Old logprobs (current policy snapshot) + reference logprobs for KL
            logits_old = policy(tokens_in)[0]               # [B, T-1, V]
            old_logprobs = logprobs_from_logits(logits_old, tokens_out)        # [B, T-1]

            ref_logits = ref_policy(tokens_in)[0]           # [B, T-1, V]
            ref_logprobs = logprobs_from_logits(ref_logits, tokens_out)  
            with torch.no_grad():
             kl_per_token = (old_logprobs - ref_logprobs)
             batch_kl = (kl_per_token * gen_mask).sum() / gen_mask.sum().clamp_min(1.0)
             if batch_kl > 1.5 * target_kl:
               kl_coef *= 1.5
             elif batch_kl < 0.5 * target_kl:
               kl_coef /= 1.5
      # [B, T-1]

            # Reward model score per sequence (scalar)
            rm_score = reward_model(sequences).mean(dim=1)  
            # (optional) normalize per batch
            

            # Token-level returns: broadcast RM score across generated tokens, minus KL
            
            kl_token = (old_logprobs - ref_logprobs)     # [B, T-1]
            rm_per_token = rm_score.unsqueeze(1).expand_as(old_logprobs)  # [B, T-1]
            returns = rm_per_token - kl_coef * kl_token
            returns = returns * gen_mask

            # Baseline: per-sample mean over generated region
            denom = gen_mask.sum(dim=1).clamp_min(1.0)
            baseline = (returns.sum(dim=1) / denom).unsqueeze(1)  # [B,1]
            advantages = (returns - baseline) * gen_mask

            # Advantage normalization across all valid tokens in batch
            adv_mean = (advantages * gen_mask).sum() / gen_mask.sum().clamp_min(1.0)
            adv_var  = ((advantages - adv_mean)**2 * gen_mask).sum() / gen_mask.sum().clamp_min(1.0)
            advantages = (advantages - adv_mean) / torch.sqrt(adv_var + 1e-8)

            # Detach "old" tensors for PPO epochs
            old_logprobs = old_logprobs.detach()
            advantages   = advantages.detach()
            gen_mask     = gen_mask.detach()

        # --- PPO updates (recompute NEW logprobs each inner epoch) ---
        batch_loss = 0.0
        for _ in range(ppo_epochs):
            logits_new = policy(tokens_in)[0]                           # [B, T-1, V]
            new_logprobs = logprobs_from_logits(logits_new, tokens_out)

            # Policy loss (only generated tokens)
            loss_policy = ppo_clip_loss(old_logprobs, new_logprobs, advantages, gen_mask, eps_clip=eps_clip)

            # Entropy bonus (encourage exploration on generated region)
            ent = entropy_from_logits(logits_new)                    # [B, T-1]
            entropy_loss = -entropy_coef * masked_mean(ent, gen_mask)

            loss = loss_policy + entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            batch_loss += loss.item()

        global_step += 1
        avg_loss = batch_loss / max(ppo_epochs, 1)
        print(f"epoch {epoch} | step {global_step} | loss {avg_loss:.4f} | rm_mean {rm_score.mean().item():+.3f}")
        pass

    # === Epoch-end evaluation ===
    evaluate_after_epoch(epoch, policy, tokenizer, reward_model)
    policy.train() 

# Save PPO-tuned policy
torch.save(policy.state_dict(), "ppo_model.pt")
print("Training done. Saved to ppo_model.pt")


