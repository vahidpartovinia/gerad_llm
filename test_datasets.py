import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Load the binary files
train_arr = np.memmap('data/mmlu/train.bin', dtype=np.uint16, mode='r')
val_arr = np.memmap('data/mmlu/val.bin', dtype=np.uint16, mode='r')

# Decode first 1000 tokens to see structure
print("TRAIN DATA:")
print(enc.decode(train_arr[:1000]))
print("\n" + "="*50 + "\n")

print("VAL DATA:")
print(enc.decode(val_arr[:1000]))