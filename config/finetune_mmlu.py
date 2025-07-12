import time

out_dir = 'out-mmlu'
eval_interval = 2
eval_iters = 10
wandb_log = False # feel free to turn on
wandb_project = 'mmlu'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'mmlu'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 4

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

