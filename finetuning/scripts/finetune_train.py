import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = "../finetune_config.yaml"
with open(config_path, "r") as f:
    finetune_configs_yaml = yaml.safe_load(f)
selected_dataset = finetune_configs_yaml["selected_dataset"]
config_dict = finetune_configs_yaml[selected_dataset]

class Config:
    def __init__(self, d):
        self.__dict__.update(d)

config = Config(config_dict)

# Ensure numeric hyperparameters are the correct type
config.learning_rate = float(config.learning_rate)
config.batch_size = int(config.batch_size)
config.epochs = int(config.epochs)
config.eval_interval = int(config.eval_interval)
config.max_seq_length = int(config.max_seq_length)

# Import model classes
from pretraining.scripts.model import GPT
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from finetuning.models.nano_gpt_classifier import NanoGPTClassifier
from finetuning.models.gpt2_classifier import GPT2Classifier, gpt2_collate_fn

# Example dataset loaders (expand as needed)
class SST2Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_length, return_text=False):
        self.samples = []
        self.return_text = return_text
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        for _, row in dataframe.iterrows():
            text = row['sentence']
            label = int(row['label'])
            if return_text:
                self.samples.append((text, label))
            else:
                tokens = tokenizer(text, max_length=max_seq_length)
                self.samples.append((tokens, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

class MMLUDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_seq_length):
        import pandas as pd
        file_path = os.path.join(data_dir, 'test.csv')
        df = pd.read_csv(file_path)
        self.samples = []
        for _, row in df.iterrows():
            prompt = row['prompt']
            choices = [row['A'], row['B'], row['C'], row['D']]
            answer = row['answer']
            text = f"Question: {prompt}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
            tokens = tokenizer(text, max_length=max_seq_length)
            self.samples.append((tokens, answer))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# Tokenizer wrappers
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

# Main fine-tuning function
def main():
    # Select config using selected_dataset from config file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Fine-tuning model: {config.model_type} on dataset: {config.dataset}")

    # Load tokenizer and model
    if config.model_type == "nano-gpt":
        # Load character-level vocab from Shakespeare dataset
        from pretraining.scripts.pretrain_data import ShakespeareDataset
        shakespeare_data_dir = os.path.join(PROJECT_ROOT, "data")
        shakespeare_dataset = ShakespeareDataset(shakespeare_data_dir, block_size=128, download=False)
        vocab = shakespeare_dataset.stoi
        tokenizer = NanoGPTTokenizer(vocab)
        # Load model config
        with open("../../pretraining/pretrain_config.yaml", "r") as f:
            pretrain_configs = yaml.safe_load(f)
        nano_config_dict = pretrain_configs["nano"]
        class Config:
            def __init__(self, d):
                self.__dict__.update(d)
        nano_config = Config(nano_config_dict)
        nano_config.vocab_size = len(vocab)
        # Use classifier for SST-2, base GPT for others
        if config.dataset == "sst2":
            model = NanoGPTClassifier(nano_config, num_classes=2)
        else:
            model = GPT(nano_config)
        # Load pre-trained weights if available
        ckpt_path = os.path.join(PROJECT_ROOT, "pretraining", "checkpoints", "nano_gpt_shakespeare.pt")
        if os.path.exists(ckpt_path):
            if config.dataset == "sst2":
                model.gpt.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
    elif config.model_type == "gpt2":
        model = GPT2Classifier(config.gpt2_dir, num_classes=2)
        tokenizer = model.tokenizer
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    # Use absolute data_dir paths
    config_data_dir = os.path.join(PROJECT_ROOT, config.data_dir)

    # Load dataset
    if config.dataset == "sst2":
        import pandas as pd
        df = pd.read_csv(os.path.join(config_data_dir, "train.tsv"), sep="\t")
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["label"]
        )
        if config.model_type == "gpt2":
            train_dataset = SST2Dataset(train_df, tokenizer, config.max_seq_length, return_text=True)
            val_dataset = SST2Dataset(val_df, tokenizer, config.max_seq_length, return_text=True)
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True,
                collate_fn=lambda batch: gpt2_collate_fn(batch, tokenizer, config.max_seq_length)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False,
                collate_fn=lambda batch: gpt2_collate_fn(batch, tokenizer, config.max_seq_length)
            )
        else:
            train_dataset = SST2Dataset(train_df, tokenizer, config.max_seq_length)
            val_dataset = SST2Dataset(val_df, tokenizer, config.max_seq_length)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    elif config.dataset == "mmlu":
        dataset = MMLUDataset(config_data_dir, tokenizer, config.max_seq_length)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = None
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        train_losses = []
        train_correct = 0
        train_total = 0
        for batch in pbar:
            if config.model_type == "nano-gpt":
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                if config.dataset == "sst2":
                    logits, loss = model(inputs, labels)
                else:
                    logits, loss = model(inputs, labels)
            elif config.model_type == "gpt2":
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits, loss = model(inputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2%}")
        # Validation after each epoch (for SST-2)
        if config.dataset == "sst2":
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    if config.model_type == "nano-gpt":
                        logits, loss = model(inputs, labels)
                    elif config.model_type == "gpt2":
                        logits, loss = model(inputs, labels)
                    val_losses.append(loss.item())
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_loss = sum(val_losses) / len(val_losses)
            val_acc = correct / total if total > 0 else 0.0
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2%}")
            model.train()
        # Save model after each epoch
        os.makedirs(config.save_dir, exist_ok=True)
        epoch_save_path = os.path.join(
            config.save_dir,
            f"{os.path.splitext(config.model_save_name)[0]}_epoch{epoch+1}.pt"
        )
        if config.model_type == "nano-gpt" and config.dataset == "sst2":
            torch.save(model.state_dict(), epoch_save_path)
        elif config.model_type == "nano-gpt":
            torch.save(model.state_dict(), epoch_save_path)
        else:
            torch.save(model.state_dict(), epoch_save_path)
        print(f"Model saved to {epoch_save_path}")

if __name__ == "__main__":
    main() 