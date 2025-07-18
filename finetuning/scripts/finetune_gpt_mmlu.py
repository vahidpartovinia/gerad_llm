 import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import requests
from pathlib import Path

from model import GPT
from config import nano_config
from data import ShakespeareDataset

class MMLUDataset(Dataset):
    def __init__(self, data_dir, split='dev', download=True):
        self.data_dir = data_dir
        self.split = split
        
        if download:
            self.download_mmlu()
        
        # Load the dataset
        self.data = self.load_data()
        
        # Create character-level encoding
        self.stoi = {}  # Will be populated from the pre-trained model
        self.itos = {}  # Will be populated from the pre-trained model
    
    def download_mmlu(self):
        """Download MMLU dataset if not already present"""
        os.makedirs(self.data_dir, exist_ok=True)
        base_url = "https://raw.githubusercontent.com/hendrycks/test/master/data"
        splits = {
            'dev': 'dev.json',
            'val': 'val.json',
            'test': 'test.json'
        }
        
        for split, filename in splits.items():
            # Use the correct URL format for MMLU
            url = f"https://raw.githubusercontent.com/hendrycks/test/master/data/{split}/{filename}"
            print(f"Downloading {split} split from {url}...")
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Save as .json file
                with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Successfully downloaded {split} split")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {split} split: {e}")
                continue
    
    def load_data(self):
        """Load and preprocess MMLU data"""
        file_path = os.path.join(self.data_dir, f'{self.split}.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MMLU {self.split} split not found at {file_path}")
        
        # Load and process the file line by line
        processed_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Try to parse each line as a separate JSON object
                    item = json.loads(line)
                    
                    # Extract the required fields
                    question = item.get('question', '')
                    choices = item.get('choices', [])
                    answer = item.get('answer', '')
                    
                    if not all([question, choices, answer]):
                        print(f"Warning: Skipping item with missing fields")
                        continue
                    
                    # Format as: "Question: [question]\nA) [choice A]\nB) [choice B]\nC) [choice C]\nD) [choice D]\nAnswer: [answer]"
                    formatted_text = f"Question: {question}\n"
                    for i, choice in enumerate(choices):
                        formatted_text += f"{chr(65+i)}) {choice}\n"
                    formatted_text += f"Answer: {answer}\n"
                    
                    processed_data.append(formatted_text)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error processing line: {e}")
                    continue
        
        if not processed_data:
            raise ValueError(f"No valid data found in {file_path}")
        
        print(f"Successfully loaded {len(processed_data)} questions from {self.split} split")
        return processed_data
    
    def set_vocabulary(self, stoi, itos):
        """Set vocabulary from pre-trained model"""
        self.stoi = stoi
        self.itos = itos
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        # Convert text to token indices
        tokens = [self.stoi.get(c, 0) for c in text]  # Use 0 for unknown characters
        return torch.tensor(tokens, dtype=torch.long)

def main():
    # Training hyperparameters
    batch_size = 32  # Smaller batch size for fine-tuning
    learning_rate = 1e-5  # Lower learning rate for fine-tuning
    max_iters = 1000
    eval_interval = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model_path = "nano_gpt_shakespeare.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
    
    # Load the dataset to get vocabulary
    shakespeare_dataset = ShakespeareDataset("data", nano_config.block_size)
    nano_config.vocab_size = shakespeare_dataset.vocab_size
    
    # Create and load model
    model = GPT(nano_config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Create MMLU dataset
    print("Loading MMLU dataset...")
    mmlu_dataset = MMLUDataset("data/mmlu", split='dev')
    mmlu_dataset.set_vocabulary(shakespeare_dataset.stoi, shakespeare_dataset.itos)
    
    # Create data loader
    train_loader = DataLoader(mmlu_dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer with lower learning rate
    optimizer = model.configure_optimizers(
        weight_decay=0.01,  # Reduced weight decay for fine-tuning
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )
    
    # Training loop
    print("Starting fine-tuning...")
    pbar = tqdm(range(max_iters), desc="Fine-tuning")
    for iter in pbar:
        # Get batch
        batch = next(iter(train_loader))
        batch = batch.to(device)
        
        # Forward pass
        logits, loss = model(batch[:, :-1], batch[:, 1:])
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'gpu_mem': f'{torch.cuda.memory_allocated()/1024**2:.0f}MB'
        })
        
        # Evaluate
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(10):  # Evaluate on 10 batches
                    batch = next(iter(train_loader))
                    batch = batch.to(device)
                    _, loss = model(batch[:, :-1], batch[:, 1:])
                    eval_losses.append(loss.item())
                eval_loss = sum(eval_losses) / len(eval_losses)
                print(f"\nIter {iter}: eval loss {eval_loss:.4f}")
            model.train()
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), "nano_gpt_mmlu.pt")
    print("\nFine-tuning completed!")
    print(f"Model saved as 'nano_gpt_mmlu.pt'")

if __name__ == "__main__":
    main() 