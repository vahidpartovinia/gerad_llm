## Data Directory
In each subdirectory, you will find the following file:
`prepare.py`. This is a script that takes the dataset and prepares it for training.
<br>
<br>
Depending on the task it will provide the different splits that are needed for the corrospodnding
training file. For example, the `prepare.py` script in the `shakespeare` directory will prepare the raw shakespeare text
as `input.txt` but it will also prepare the `train.bin` and `val.bin` files that are used for training and validation.

