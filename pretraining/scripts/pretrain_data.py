import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, block_size, download=True):
        self.block_size = block_size
        self.data_dir = data_dir
        
        if download:
            self.download_shakespeare()
        
        # Read the data
        with open(os.path.join(data_dir, 'shakespeare.txt'), 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Create character-level encoding
        chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Encode the text
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)
    
    def download_shakespeare(self):
        """Download Shakespeare's complete works if not already present"""
        os.makedirs(self.data_dir, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(os.path.join(self.data_dir, 'shakespeare.txt'), 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Grab a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class ShakespeareDataLoader:
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.data = dataset.data
        self.block_size = dataset.block_size
        
        # Pre-compute all possible starting indices
        self.indices = torch.arange(len(dataset))
        
    def get_batch(self):
        """Generate a batch of data efficiently"""
        # Randomly sample batch_size indices
        idx = torch.randint(len(self.dataset), (self.batch_size,))
        
        # Get all sequences at once
        x = torch.stack([self.data[i:i+self.block_size] for i in idx])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in idx])
        
        # Move to device
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_eval_batch(self, num_samples):
        """Generate a large batch for evaluation"""
        # Use sequential indices for evaluation to ensure coverage
        idx = torch.arange(num_samples) % len(self.dataset)
        
        # Get all sequences at once
        x = torch.stack([self.data[i:i+self.block_size] for i in idx])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in idx])
        
        # Move to device
        x, y = x.to(self.device), y.to(self.device)
        return x, y 