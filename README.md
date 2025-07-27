# <u> Using NanoGPT for ML pipeline </u>

All training files and configuration files work together to begin training.
Commands usually follow this format:
```
python [training file.py] config/[config file].py
```
You may also add --[override] to change parameters in the config file. For example:
```
python train.py config/train_shakespeare_char.py --device=cpu --batch_size=12
```
Note: You may either use python or python3 depending on your system configuration.
