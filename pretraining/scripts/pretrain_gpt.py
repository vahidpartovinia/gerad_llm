import sys
import os
import yaml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = "../../pretraining/pretrain_config.yaml"
with open(config_path, "r") as f:
    pretrain_configs = yaml.safe_load(f)
nano_config_dict = pretrain_configs["nano"]

class Config:
    def __init__(self, d):
        self.__dict__.update(d)
        # Default to normal if not specified
        if not hasattr(self, 'weight_init'):
            self.weight_init = 'normal'

nano_config = Config(nano_config_dict)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretraining.scripts.model import GPT
from pretraining.scripts.pretrain_data import ShakespeareDataset, ShakespeareDataLoader

# Training hyperparameters
batch_size = 64  # Reduced from 512 to fit in 4GB VRAM
learning_rate = 3e-4
max_iters = 5000
eval_interval = 100  # Evaluate every 100 training iterations
eval_samples = 256  # Reduced from 2048 to fit in 4GB VRAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_wandb = False  # Set to True if you want to use W&B

def main():
    # Initialize wandb if enabled
    if use_wandb:
        import wandb
        wandb.init(project="nano-gpt-shakespeare", config={
            "model": "nano-gpt",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
        })

    # Create data directory and dataset
    data_dir = "../../data"
    dataset = ShakespeareDataset(data_dir, nano_config.block_size)
    
    # Create optimized data loader
    data_loader = ShakespeareDataLoader(dataset, batch_size, device)
    
    # Create model with correct vocabulary size
    nano_config.vocab_size = dataset.vocab_size  # Use actual vocabulary size from dataset
    model = GPT(nano_config)
    model = model.to(device)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total characters in dataset: {len(dataset.text)}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Unique characters: {len(dataset.stoi)}")
    print(f"Context window size: {nano_config.block_size}")
    
    # Print model information
    print("\nModel Information:")
    print(f"Number of layers: {nano_config.n_layer}")
    print(f"Number of attention heads: {nano_config.n_head}")
    print(f"Embedding dimension: {nano_config.n_embd}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Print training information
    print("\nTraining Information:")
    print(f"Batch size: {batch_size} sequences")
    print(f"Learning rate: {learning_rate}")
    print(f"Total iterations: {max_iters}")
    print(f"Device: {device}")
    print(f"Evaluation interval: every {eval_interval} iterations")
    print(f"Evaluation samples: {eval_samples} sequences")
    
    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )

    # Training loop with progress bar
    pbar = tqdm(range(max_iters), desc="Training")
    for iter in pbar:
        # Get batch using optimized loader
        xb, yb = data_loader.get_batch()
        
        # Forward pass
        logits, loss = model(xb, yb)
        
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
        
        # Log metrics
        if iter % 10 == 0:
            if use_wandb:
                wandb.log({
                    "iter": iter,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Evaluate
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get evaluation data using optimized loader
                eval_losses = []
                eval_pbar = tqdm(range(eval_samples // batch_size), desc="Evaluating")
                for _ in eval_pbar:
                    xb, yb = data_loader.get_eval_batch(batch_size)
                    _, loss = model(xb, yb)
                    eval_losses.append(loss.item())
                    eval_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                eval_loss = sum(eval_losses) / len(eval_losses)
                print(f"\nIter {iter}: eval loss {eval_loss:.4f}")
                if use_wandb:
                    wandb.log({"eval_loss": eval_loss})
            model.train()
        
        # Generate sample text
        if iter % 500 == 0:
            model.eval()
            with torch.no_grad():
                # Get a random starting context from the dataset
                start_idx = torch.randint(0, len(dataset), (1,)).item()
                x, _ = dataset[start_idx]  # Get a single sequence
                context = x.unsqueeze(0).to(device)  # Add batch dimension
                
                # Generate with temperature control
                temperature = 0.8  # Lower temperature = more focused/safer predictions
                x = model.generate(context, max_new_tokens=100, temperature=temperature)[0]
                
                # Convert the starting context to text
                start_text = ''.join([dataset.itos[int(i)] for i in context[0]])
                # Convert the generated continuation to text
                completion = ''
                for i in x[len(context[0]):]:  # Only show the newly generated part
                    token_id = int(i)
                    if token_id in dataset.itos:
                        completion += dataset.itos[token_id]
                    else:
                        completion += '?'  # Use '?' for unknown tokens
                
                print(f"\nSample generation at iter {iter}:")
                print(f"Context: {start_text}")
                print(f"Generated: {completion}\n")
                if use_wandb:
                    wandb.log({
                        "sample_generation": wandb.Html(f"Context: {start_text}<br>Generated: {completion}")
                    })
            model.train()

    # Save the model
    torch.save(model.state_dict(), "nano_gpt_shakespeare_xavier.pt")
    if use_wandb:
        wandb.save("nano_gpt_shakespeare.pt")
    
    print("\nTraining completed!")
    print(f"Model saved as 'nano_gpt_shakespeare_xavier.pt'")

if __name__ == "__main__":
    main() 