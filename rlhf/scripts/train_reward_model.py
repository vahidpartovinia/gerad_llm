import sys
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from gerad_llm.rlhf.models.reward_model import RewardModel

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'rlhf_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config_yaml = yaml.safe_load(f)
reward_config = config_yaml['reward_model']

# Ensure correct types for hyperparameters
reward_config['learning_rate'] = float(reward_config['learning_rate'])
reward_config['batch_size'] = int(reward_config['batch_size'])
reward_config['epochs'] = int(reward_config['epochs'])
max_samples = int(reward_config.get('max_samples', 0)) if 'max_samples' in reward_config else None

# Download Anthropic HH-RLHF dataset if not present
from pathlib import Path
import datasets

def ensure_hh_rlhf_dataset(local_path, split):
    if not Path(local_path).exists():
        print(f"Downloading Anthropic HH-RLHF {split} split...")
        ds = datasets.load_dataset('Anthropic/hh-rlhf', split=split)
        with open(local_path, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {split} split to {local_path}")
    else:
        print(f"Found existing {split} split at {local_path}")

ensure_hh_rlhf_dataset(reward_config['dataset_path'], 'train')
ensure_hh_rlhf_dataset(reward_config['val_dataset_path'], 'test')

# Dataset for Anthropic HH-RLHF pairs
class AnthropicPairDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, block_size=128, max_samples=None):
        import json
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get('prompt', '')
                chosen = item['chosen']
                rejected = item['rejected']
                self.samples.append((prompt, chosen, rejected))
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.tokenizer = tokenizer
        self.block_size = block_size
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        prompt, chosen, rejected = self.samples[idx]
        # Concatenate prompt and response, truncate/pad
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        chosen_ids = self.tokenizer(chosen_text, max_length=self.block_size)
        rejected_ids = self.tokenizer(rejected_text, max_length=self.block_size)
        return chosen_ids, rejected_ids

# Tokenizer for NanoGPT (character-level)
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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training reward model on device: {device}")

    model_type = reward_config.get('model_type', 'nano-gpt')
    print(f"Using base model: {model_type}")

    if model_type == 'nano-gpt':
        from gerad_llm.pretraining.scripts.model import GPT
        from gerad_llm.pretraining.scripts.pretrain_data import ShakespeareDataset
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        shakespeare_dataset = ShakespeareDataset(data_dir, block_size=128, download=False)
        vocab = shakespeare_dataset.stoi
        tokenizer = NanoGPTTokenizer(vocab)
        # Load GPT config
        pretrain_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pretraining', 'pretrain_config.yaml'))
        with open(pretrain_config_path, 'r') as f:
            pretrain_configs = yaml.safe_load(f)
        nano_config_dict = pretrain_configs['nano']
        class Config:
            def __init__(self, d):
                self.__dict__.update(d)
        nano_config = Config(nano_config_dict)
        nano_config.vocab_size = len(vocab)
        # Load base GPT model
        base_model = GPT(nano_config)
        # Load the checkpoint
        ckpt = torch.load(reward_config['pretrained_checkpoint'], map_location=device)
        # If checkpoint is from NanoGPTClassifier, extract the 'gpt.' submodule and strip prefix
        # This is necessary because the classifier wrapper saves weights as 'gpt.*', but GPT expects no prefix
        if any(k.startswith('gpt.') for k in ckpt.keys()):
            new_ckpt = {k.replace('gpt.', ''): v for k, v in ckpt.items() if k.startswith('gpt.')}
            ckpt = new_ckpt
        base_model.load_state_dict(ckpt, strict=False)
        base_model = base_model.to(device)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
    elif model_type == 'gpt2':
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        base_model = base_model.to(device)
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        hf_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        def gpt2_tokenizer(text, max_length, **kwargs):
            tokens = hf_tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')['input_ids'].squeeze(0)
            return tokens
        tokenizer = gpt2_tokenizer
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create reward model
    reward_model = RewardModel(base_model)
    reward_model = reward_model.to(device)

    # Dataset and DataLoader
    train_dataset = AnthropicPairDataset(reward_config['dataset_path'], tokenizer, block_size=128, max_samples=max_samples)
    val_dataset = AnthropicPairDataset(reward_config['val_dataset_path'], tokenizer, block_size=128, max_samples=max_samples)
    train_loader = DataLoader(train_dataset, batch_size=reward_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=reward_config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=reward_config['learning_rate'])

    # Training loop
    for epoch in range(reward_config['epochs']):
        reward_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{reward_config['epochs']}")
        train_losses = []
        for chosen_ids, rejected_ids in pbar:
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            chosen_reward = reward_model(chosen_ids)
            rejected_reward = reward_model(rejected_ids)
            loss = reward_model.loss(chosen_reward, rejected_reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        reward_model.eval()
        val_losses = []
        with torch.no_grad():
            for chosen_ids, rejected_ids in val_loader:
                chosen_ids = chosen_ids.to(device)
                rejected_ids = rejected_ids.to(device)
                chosen_reward = reward_model(chosen_ids)
                rejected_reward = reward_model(rejected_ids)
                loss = reward_model.loss(chosen_reward, rejected_reward)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Save checkpoint
        os.makedirs(reward_config['save_dir'], exist_ok=True)
        save_path = os.path.join(reward_config['save_dir'], f"{os.path.splitext(reward_config['model_save_name'])[0]}_{model_type}_epoch{epoch+1}.pt")
        torch.save(reward_model.state_dict(), save_path)
        print(f"Reward model saved to {save_path}")

if __name__ == "__main__":
    main() 