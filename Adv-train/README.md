# Fast Adversarial Training with Adaptive Step Size

This repository contains the code for reproducing ATAS, of our paper: [Fast Adversarial Training with Adaptive Step Size](https://arxiv.org/abs/2206.02417).

I have adapted the code for GTSRB. The original code for CIFAR and ImageNet are not tested under the following environment, but they should work.

## Prerequisites
The code is tested under the following environment and it should be compatible with other versions of python and pytorch.
- python 3.12.3, CUDA 12.4
- pytorch 2.5.1, torchvision 0.20.1
- ~~Install autoattack with
```pip install git+https://github.com/fra31/auto-attack```~~


## Instructions to Run the Code

~~Please first change the ROOT in `data.py` to your data folder. Then run~~ (They are for the old code base!)

```
bash run.sh
```

Put the data under `./data` folder, e.g. `./data/GTSRB`. Then run `./run_atas_gtsrb.sh`. Also change the configuration there.

The results will be stored in `results/` with the saved models and their test accuracies. The log will be saved under `log/`

## Code Overview

The following is from the original paper.

The directory `models` contains model architecture definition files. The functions of the python files are:

- `ATAS.py`: code for training with ATAS.
- `attack.py`: code for the evaluation of our model. 
- `data.py`: data loading.
- `adv_attack.py`: generating adversarial examples for the training.
- `data_aug.py`: data augmentation and inverse data augmentation.

The functions of the scripts are
- `ATAS_CIFAR.sh`: running the training and evaluation for CIFAR10 and CIFAR100.
- `ATAS_ImageNet.sh`: running the training and evaluation for ImageNet.
- `run.sh`: example command for runing `ATAS_CIFAR.sh` and `ATAS_ImageNet.sh`

## Citation

We thank the following authors for open-sourcing the code.

```
@article{Huang2023ATAS,
    title={Fast Adversarial Training with Adaptive Step Size},
    author={Huang, Zhichao and Fan, Yanbo and Liu, Chen and Zhang, Weizhong and Zhang, Yong Zhang and Salzmann, Mathieu and Süsstrunk, Sabine and Wang, Jue},
    booktitle={IEEE Transcations on Image Processing},
    year={2023},
}
```
