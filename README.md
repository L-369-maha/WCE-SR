# WCE-SR
This repository is an official PyTorch implementation of the paper **"Multi-level Domain Adaptation for Wireless Capsule Endoscopy Image Super-Resolution"**.

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

In this project, we propose a multi-level domain adaptation training framework for the SR of capsule endoscopy images.

## 🚉: Pre-Trained Models

To achieve SR of capsule endoscopy images, download these [2x](https://drive.google.com/file/d/1eA4xI5CkbZh6Z46sfQu8RtJgGx6RrOVx/view?usp=share_link), [4x](https://drive.google.com/file/d/1WJFxgYJEt4zAhj7Lo7PuWU3_z92rz3Iw/view?usp=sharing) models
## 🚋: Training

We first train adaptive downsampling model alone for 50 epochs, and then train domain adaptation SR model together for 50 epoch.
The detailed training command as here:
```
CUDA_VISIBLE_DEVICE=0 python train.py --name {EXP_PATH} --scale {SCALE} --adv_w 0.01 --batch_size 10 --patch_size_down 256 --decay_batch_size_sr 400000 --decay_batch_size_down 50000 --epochs_sr_start 51 --gpu cuda:0 --sr_model endosr --training_type endosr --joint --save_results --save_log
```
with following options:
- `EXP_PATH` is the folder name of experiment results
- `scale` is the scale of the SR
- `adv_w` is th hyperparameter. (default: `0.01)

## 🧩: Evaluation

The detailed evaluation command as here:
```
CUDA_VISIBLE_DEVICE=0 python predict.py --test_mode sr --name ADL_EndoSR_withoutWL/adl_endosr_x4 --scale 4 --crop 336 --pretrain_sr ./experiment/ADL_EndoSR_withoutWL/adl_endosr_x8/models/model_sr_last.pth --test_lr Capsule_Data/TestSet/Capsule_dataset02 --gpu cuda:0 --sr_model endosr --training_type endosr --save_results --realsr
```

## 🔥: E-Mail: Contact

If you have any question, please email `2027194393@qq.com`.
