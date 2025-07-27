## Models directory

In this directory, you will find the model architectures that can be used for the different tasks.
<br>
<br>
1. `model.py` is NanoGPT's default model architecture, which is a simple transformer model. It is in the `shakespeare` directory.
There is also it's `model_lora.py` counterpart, which implements the LoRA (Low-Rank Adaptation) technique. <b> Must be used with
train.py or train_lora.py scripts </b>.
<br>
<br>
2. In the `mmlu & sst2` directory, you will find `model_classification.py` and `model_classification.py`, which are the altered
versions of the NanoGPT model for classification tasks (added classification head and num_heads parameter which can be changed
from 2 to 4). <b> Must be used with
train_classification.py or train_classification_lora.py scripts </b>.
<br>
<br>
3. Finally, in the `rlhf` directory, you will find `model_classification_rlhf.py`, which is the altered
version of the NanoGPT model for RLHF tasks (added classification head and reward head).
<br>
<br>
The training for the RLHF tasks are the only ones that are not run in unison with others using command line
arguments.
<br>
<br>
Instead, you will just run the `train_reward.py` and `train_ppo.py` scripts, which will take care of the training (and model usage) in the one file.
Changing parameters such as batch_size, gradient_accumulation_steps, and learning rate can be done in the trainig file.