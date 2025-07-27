import time

out_dir = 'out-sst2'
eval_interval = 20
eval_iters = 10
wandb_log = False # feel free to turn on
wandb_project = 'sst2'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'sst2'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 4
gradient_accumulation_steps = 4

# finetune at constant LR
learning_rate = 2e-5
decay_lr = False
