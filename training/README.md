## Training directory

In this directory, you will find the main training scripts used for training the models. The scripts are designed to be run from the command line (apart from the rlhf ones), and they will use the configuration files found in the `config` directory to set the parameters for training.

### Usage Example
Much like the example provided in the root `README.md`, you can run the training script with a command like this:

```
python3 train_classification_lora.py config/finetune_sst2.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=256 --batch_size=4 --n_layer=8 --n_head=4 --n_embd=512 --max_iters=3000 --lr_decay_iters=3000 --dropout=0.1 --init_from=resume
```
This command will start training a classification model using the LoRA adapter on the SST-2 dataset, with various parameters set according to the configuration file and command line arguments.