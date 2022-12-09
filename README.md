# Pruning-Defending-Against-Backdooring-Attack
 
## Requirements
Any version of
- hp5y, tensorflow, numpy, matplotlib
And
- python3.6+

## How to train
```
# Easily start a new training, run: 
python train_and_prune.py

# You can manually assign location of your dataset and model with: 
python train_and_prune.py --clean_valid="data/cl/valid.h5" --clean_test="data/cl/test.h5" --bad_test="data/bd/bd_test.h5" --bad_net="models/bd_net.h5"

# To list all configurable parameters use: 
python train_and_prune.py -h
```

## How to test your pruning defense
```
# Easily start a new testing by indicating your models save path and run: 
python test_robustness.py --save_path="models/repaired/"
```