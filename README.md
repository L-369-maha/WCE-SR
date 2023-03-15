# SR

## Dependencies
* Python 3.7
* PyTorch >= 1.7.0
* matplotlib
* yaml
* importlib
* functools
* scipy
* numpy
* tqdm
* PIL


## 🚉: Pre-Trained Models

download these [2x], [4x] models
## 🚋: Training

The detailed training command as here:
```
CUDA_VISIBLE_DEVICE=0 python train.py
```
with following options:
- `EXP_PATH` is the folder name of experiment results
- `scale` is the scale of the SR
- `adv_w` is th hyperparameter. (default: `0.01)

## 🧩: Evaluation

The detailed evaluation command as here:
```
CUDA_VISIBLE_DEVICE=0 python predict.py
```
