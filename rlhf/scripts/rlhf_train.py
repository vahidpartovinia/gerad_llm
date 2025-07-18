import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import random
from gerad_llm.pretraining.scripts.model import GPT
from gerad_llm.pretraining.scripts.pretrain_data import ShakespeareDataset
from gerad_llm.rlhf.models.reward_model import RewardModel

# Helper to get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))

# Load config
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'gerad_llm', 'rlhf', 'configs', 'rlhf_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config_yaml = yaml.safe_load(f)
rlhf_config = config_yaml['rlhf']

# Load NanoGPT config and vocab
PRETRAIN_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'gerad_llm', 'pretraining', 'pretrain_config.yaml')
with open(PRETRAIN_CONFIG_PATH, 'r') as f:
    pretrain_configs = yaml.safe_load(f)
nano_config_dict = pretrain_configs['nano']
class Config:
    def __init__(self, d):
        self.__dict__.update(d)
nano_config = Config(nano_config_dict)

SHAKESPEARE_DATA_DIR = os.path.join(PROJECT_ROOT, 'gerad_llm', 'data')
shakespeare_dataset = ShakespeareDataset(SHAKESPEARE_DATA_DIR, block_size=nano_config.block_size, download=False)
vocab = shakespeare_dataset.stoi
nano_config.vocab_size = len(vocab)

# Tokenizer
class NanoGPTTokenizer:
    def __init__(self, vocab):
        self.stoi = vocab
    def __call__(self, text, max_length):
        tokens = [self.stoi.get(c, 0) for c in text]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += [0] * (max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
tokenizer = NanoGPTTokenizer(vocab)

# Load NanoGPT model
PRETRAINED_CKPT_PATH = os.path.join(PROJECT_ROOT, rlhf_config['pretrained_checkpoint'])
model = GPT(nano_config)
ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location='cpu')
if any(k.startswith('gpt.') for k in ckpt.keys()):
    new_ckpt = {k.replace('gpt.', ''): v for k, v in ckpt.items() if k.startswith('gpt.')}
    ckpt = new_ckpt
model.load_state_dict(ckpt, strict=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.train()

# Load reward model
REWARD_MODEL_CKPT_PATH = os.path.join(PROJECT_ROOT, rlhf_config['reward_model_checkpoint'])
reward_model = RewardModel(model)
reward_model.load_state_dict(torch.load(REWARD_MODEL_CKPT_PATH, map_location='cpu'))
reward_model = reward_model.to(device)
reward_model.eval()

# PPO hyperparameters
ppo_epochs = int(rlhf_config.get('epochs', 1))
batch_size = int(rlhf_config.get('batch_size', 8))
max_length = int(rlhf_config.get('max_length', 128)) if 'max_length' in rlhf_config else 128
clip_epsilon = 0.2
learning_rate = float(rlhf_config.get('learning_rate', 1e-5))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prompts (for demo, use a small set or sample from data)
prompts = [
    "How do I bake a cake?",
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "What is the meaning of life?",
    "How do I train a neural network?",
    "Tell me a joke.",
    "What is quantum computing?",
    "How do I improve my memory?"
]

# Helper: Generate response from NanoGPT
def generate_response(model, prompt, tokenizer, max_length=128, device='cpu'):
    model.eval()
    tokens = tokenizer(prompt, max_length)
    input_ids = tokens.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(32):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.shape[1] >= max_length:
                break
    return input_ids[0]

# Helper: Compute logprobs for a sequence
def compute_logprobs(model, input_ids):
    logits, _ = model(input_ids.unsqueeze(0))
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = log_probs[0, range(input_ids.size(0)), input_ids]
    return token_logprobs

for epoch in range(ppo_epochs):
    random.shuffle(prompts)
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start+batch_size]
        batch_input_ids = []
        batch_rewards = []
        batch_old_logprobs = []

        # 1. Generate responses and compute rewards
        for prompt in batch_prompts:
            response_ids = generate_response(model, prompt, tokenizer, max_length, device)
            batch_input_ids.append(response_ids)
            # Compute reward
            text = prompt + ''.join([shakespeare_dataset.itos[int(i)] for i in response_ids])
            input_ids = tokenizer(text, max_length).unsqueeze(0).to(device)
            with torch.no_grad():
                reward = reward_model(input_ids)
            batch_rewards.append(reward.item())
            # Compute old logprobs
            old_logprobs = compute_logprobs(model, response_ids.to(device))
            batch_old_logprobs.append(old_logprobs.detach())

        # 2. PPO update
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for i, response_ids in enumerate(batch_input_ids):
            logprobs = compute_logprobs(model, response_ids.to(device))
            old_logprobs = batch_old_logprobs[i]
            reward = batch_rewards[i]
            ratio = torch.exp(logprobs - old_logprobs)
            advantage = reward  # For demo, use reward as advantage
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
            loss = -torch.min(surr1, surr2).mean()
            total_loss += loss
        total_loss /= len(batch_input_ids)
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {batch_start//batch_size+1}, Loss: {total_loss.item():.4f}")

print("PPO training complete.") 