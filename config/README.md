## <u> Using Config Files </u>
Config files are used to set the parameters for training a model. They will be the final settings unless you
override them using command line arguments.

### <u> Config File Structure </u>
```
out_dir = 'out-shakespeare' # directory to store the training output
eval_interval = 5 # how many steps to take before evaluating the model
eval_iters = 40 # how many batches to use for each evaluation
```
<u> <b> out_dir </b> </u> = the directory where the models training output will be stored.
<br>
<u> <b> eval_interval </b> </u> = the number of steps to take before evaluating the model.
<br>
<u> <b> eval_iters </b> </u> = number of batches to use for each evaluation.
<br>

```
dataset = 'shakespeare'
init_from = 'gpt2-xl'
```
<u> <b> dataset </b> </u> = the dataset to use for training, must match the name of the subdirectory in the `data` directory. For example,
if you have a dataset in `data/mmlu`, you would set this to `mmlu`.
<br>
<u> <b> init_from </b> </u> = the model to use for training. One of the ones that you may override using --init_from=[option].
There are a few options available, such as the gpt2's found in the 'models' directory, or you may use --init_from=resume or
--init_from=scratch to resume training from a checkpoint or start from scratch, respectively.
<br>

```
always_save_checkpoint = False
```
<u> <b> always_save_checkpoint </b> </u> = whether to always save a checkpoint after each evaluation. If set to `True`, it will save
a checkpoint after each evaluation, which can be useful for long training runs. If set to `False`, it will only save a checkpoint
after the best evaluation score is achieved.
<br>

```
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20
``` 
<u> <b> batch_size </b> </u> = the batch size to use for training.
<br>
<u> <b> gradient_accumulation_steps </b> </u> = the number of steps to accumulate gradients before updating the model weights. Can be
useful to simulate larger batch sizes without running out of memory.
<br>
<u> <b> max_iters </b> </u> = the maximum number of iterations to train for.
<br>

```
learning_rate = 3e-5
decay_lr = False
```
<u> <b> learning_rate </b> </u> = the learning rate to use for training.
<br>
<u> <b> decay_lr </b> </u> = whether to decay the learning rate during training.
