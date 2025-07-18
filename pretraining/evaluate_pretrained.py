import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
from pretraining.scripts.model import GPT
from pretraining.scripts.pretrain_data import ShakespeareDataset
import yaml
from tqdm import tqdm

# Configurable paths
MODEL_PATH = "../nano_gpt_shakespeare.pt"  # or "nano_gpt_shakespeare_xavier.pt"
VAL_DATA_PATH = "../data/shakespeare.txt"  # Use the same as pretraining, or a held-out split
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "pretrain_config.yaml")
EVAL_SAMPLES = 9048  # Number of samples to use for evaluation

# Load config
with open(CONFIG_PATH, "r") as f:
    pretrain_configs = yaml.safe_load(f)
nano_config_dict = pretrain_configs["nano"]
class Config:
    def __init__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'weight_init'):
            self.weight_init = 'normal'
nano_config = Config(nano_config_dict)

def evaluate(model, dataset, batch_size=64, device='cpu', eval_samples=9048):
    model.eval()
    losses = []
    n_eval = min(eval_samples, len(dataset) - batch_size)
    with torch.no_grad():
        for i in tqdm(range(0, n_eval, batch_size), desc="Evaluating"):
            xb = []
            yb = []
            for j in range(batch_size):
                if i + j >= n_eval:
                    break
                x, y = dataset[i + j]
                xb.append(x)
                yb.append(y)
            if not xb:
                continue
            xb = torch.stack(xb).to(device)
            yb = torch.stack(yb).to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device}")
    # Load dataset
    val_dataset = ShakespeareDataset(os.path.dirname(VAL_DATA_PATH), nano_config.block_size, download=False)
    # Load model
    nano_config.vocab_size = val_dataset.vocab_size
    model = GPT(nano_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    # Evaluate
    val_loss, val_ppl = evaluate(model, val_dataset, batch_size=64, device=device, eval_samples=EVAL_SAMPLES)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation perplexity: {val_ppl:.2f}")

if __name__ == "__main__":
    main() 