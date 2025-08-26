
import torch
import torch.nn.functional as F
import string

from tokenizer import NanoGPTCharTokenizer, stoi, itos # <- import from a side-effect-free module
from model import GPT, GPTConfig, RewardModel

# ---------- Config ----------
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain the importance of photosynthesis in 2 sentences.",
    "Write a polite reply declining a meeting invite.",
    "Summarize the benefits of exercise in one sentence.",
]

MAX_NEW_TOKENS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths to checkpoints
BASE_CKPT = "reward_model.pt"      # pre-RLHF (before PPO) checkpoint
PPO_CKPT  = "ppo_model.pt"       # post-RLHF (after PPO training) checkpoint
REWARD_CKPT = "reward_model.pt"    # reward model weights

# ---------- Tokenizer ----------
# ---------- Tokenizer ----------
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # or your training tokenizer

# ensure padding token exists if your code pads with 0
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode_batch(texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"]

# infer vocab size from tokenizer (matches checkpoint's embeddings)
vocab_size = len(tokenizer)
print("Vocab size:", vocab_size)

def encode_batch(texts):
    ids = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
    max_len = max(x.size(0) for x in ids)
    padded = torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.long)]) for x in ids])
    return padded

@torch.no_grad()
def generate_text(model, prompt):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(x, max_new_tokens=MAX_NEW_TOKENS)
    # If your generate returns a plain tensor, handle that:
    ids = out.sequences[0].tolist() if hasattr(out, "sequences") else out[0].tolist()
    return tokenizer.decode(ids)

# ---------- Models ----------
def ckpt_vocab_size(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    # find embedding weight
    for k, v in sd.items():
        if k.endswith("transformer.wte.weight"):
            return v.shape[0]
    raise RuntimeError("Couldn't find transformer.wte.weight in checkpoint")

VOCAB_FROM_CKPT = ckpt_vocab_size("reward_model.pt")  # -> likely 50257
print("vocab from ckpt:", VOCAB_FROM_CKPT)
gptconf = GPTConfig(
    block_size=128,
    vocab_size=VOCAB_FROM_CKPT,  # must match checkpoint
    n_layer=12, n_head=12, n_embd=768,
    dropout=0.0, bias=True
)

base_model = GPT(gptconf).to(DEVICE)
ppo_model  = GPT(gptconf).to(DEVICE)
reward_backbone = GPT(gptconf).to(DEVICE)  # if your reward model uses same backbone
reward_model = RewardModel(reward_backbone).to(DEVICE)

# ---------- Load weights ----------
# ---------- Load weights (fixed for prefix) ----------
def load_lm_from_pref(model, ckpt_path, device, prefixes=("base_model.","policy.","model.","module.","")):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # detect which prefix actually contains transformer weights
    pref = None
    for p in prefixes:
        if (p + "transformer.wte.weight") in sd:
            pref = p
            break
    if pref is None:
        for p in prefixes:
            if any(k.startswith(p + "transformer.wte.weight") for k in sd.keys()):
                pref = p
                break
    if pref is None:
        raise RuntimeError(f"Couldn't find transformer weights in {ckpt_path}. "
                           f"Top keys: {list(sd)[:10]}")

    # strip prefix + drop heads not in GPT
    tgt_keys = set(model.state_dict().keys())
    filtered = {k[len(pref):]: v for k, v in sd.items() 
                if k.startswith(pref) and k[len(pref):] in tgt_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded LM from {ckpt_path} (prefix='{pref}')")
    print("  missing:", missing[:5], "..." if len(missing) > 5 else "")
    print("  unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")

# Use LM inside reward_model.pt as BASE_CKPT
load_lm_from_pref(base_model, "reward_model.pt", DEVICE)

# Use LM inside ppo_model.pt as PPO_CKPT
load_lm_from_pref(ppo_model, "ppo_model.pt", DEVICE)

# Load the full reward model wrapper
rm_sd = torch.load("reward_model.pt", map_location=DEVICE)
if isinstance(rm_sd, dict) and "state_dict" in rm_sd:
    rm_sd = rm_sd["state_dict"]
reward_model.load_state_dict(rm_sd, strict=False)
base_model.eval(); ppo_model.eval(); reward_model.eval()
torch.set_grad_enabled(False)

# ---------- Evaluate ----------
rows = []
for p in TEST_PROMPTS:
    out_base = generate_text(base_model, p)
    out_ppo  = generate_text(ppo_model,  p)

    ids_base = encode_batch([out_base]).to(DEVICE)
    ids_ppo  = encode_batch([out_ppo]).to(DEVICE)

    r_base = reward_model(ids_base)
    r_ppo  = reward_model(ids_ppo)

    if r_base.ndim == 2: r_base = r_base[:, -1]
    r_base = float(r_base.mean().item())
    if r_ppo.ndim == 2:  r_ppo  = r_ppo[:, -1]
    r_ppo  = float(r_ppo.mean().item())

    rows.append({
        "prompt": p,
        "before_output": out_base.replace("\n", " "),
        "after_output":  out_ppo.replace("\n", " "),
        "reward_before": r_base,
        "reward_after":  r_ppo,
    })

# ---------- Pretty print answers ----------
print("\n=================== RESULTS ===================")
for r in rows:
    print("\n-----------------------------------------------")
    print(f"Prompt:\n{r['prompt']}")
    print("\nBefore PPO output:")
    print(r["before_output"])
    print("\nAfter PPO output:")
    print(r["after_output"])
    print(f"\nRewards -> Before: {r['reward_before']:.3f} | After: {r['reward_after']:.3f} | Δ: {r['reward_after'] - r['reward_before']:+.3f}")

# ---------- Save CSV ----------
import csv
with open("eval_before_after.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

# ---------- Compact table ----------
print("\n\n| Prompt | Reward (Before) | Reward (After) | Δ |")
print("|--------|------------------|----------------|---|")
for r in rows:
    delta = r["reward_after"] - r["reward_before"]
    print(f"| {r['prompt'][:35]}... | {r['reward_before']:.3f} | {r['reward_after']:.3f} | {delta:+.3f} |")

print("\nSaved: eval_before_after.csv")



