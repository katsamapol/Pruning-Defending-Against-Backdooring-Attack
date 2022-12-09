# Pruning-Defending-Against-Backdooring-Attack
 
## Requirements
Any version of
- hp5y, tensorflow, numpy, matplotlib

## How to train
```
# Easily start a new training, run: 
python train_and_prune.py

# You can manually assign location of your dataset and model with: 
python project1_model.py --clean_valid="data/cl/valid.h5" --clean_test="data/cl/test.h5" --bad_test="data/bd/bd_test.h5" --bad_net="models/bd_net.h5"

# To list all configurable parameters use: 
python project1_model.py -h
