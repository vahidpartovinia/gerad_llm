 import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import os

def download_gpt2():
    print("Downloading GPT-2 model and tokenizer...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Save model and tokenizer
    model_path = Path("models/gpt2")
    model_path.mkdir(exist_ok=True)
    
    print("Saving model and tokenizer...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("GPT-2 model and tokenizer downloaded and saved successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"Saved to: {model_path.absolute()}")

if __name__ == "__main__":
    download_gpt2() 