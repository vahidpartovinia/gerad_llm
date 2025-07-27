import time

out_dir = 'out-mmlu-rlhf'
eval_interval = 10
eval_iters = 10
wandb_log = False # feel free to turn on
wandb_project = 'mmlu_rlhf'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'rlhf'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 4
gradient_accumulation_steps = 4

# finetune at constant LR
learning_rate = 1e-4
decay_lr = True
