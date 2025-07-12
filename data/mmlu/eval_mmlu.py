import os
import torch
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT

init_from = 'resume'
eval_only = True      # Exit after first evaluation
eval_iters = 100      # Number of validation batches to test
device = 'mps' if torch.mps.is_available() else 'cpu'
compile = False       # Disable compilation for faster loading

# Import and run the training script in eval mode
exec(open('train.py').read())