## Achieving Verifiable Robustness via Empirical Adversarial Training and Structured Pruning

### Code Structure

Environment Setup:

- Please refer to the README in both directories
- GPU: I tested with a single NVIDIA V100

Code:

- `Adv-train`: the adversarial training and pruning
  - `models/`: the model architecture. I only use `resnet.py` and `resnet4b.py` in this project. You can adjust the `in_planes` to control their sizes.
  - `ATAS*.sh`: script to train a `ResNet4b`, `ResNet4b-ultrawide` (`ResNet4b-wide` in the paper), `ResNet18` with adversarial training. `ATAS_CIFAR_Small.sh` contains scripts to train both with diffusion-model-generated dataset and the original CIFAR-10 dataset. The latter is currently commented. If you decide to run with the synthetic dataset from the diffusion model, please download with this [Link](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz) from this [repository](https://github.com/wzekai99/DM-Improves-AT) and put the `1m.npz` in the root directory of this project (outside `Adv-train/`)
  - `ATAS.py`: the core adversarial training algorithm
  - `prune_resnet18.[py, sh]`, `convert_pruned_model.py`, `verify_sparsity`: 1) code and script that prunes the `ResNet18` with unstructured pruning ("failure cases"); 2) convert the model file with pruning mask into a pure model where the sparsity is applied (zeroing out the corresponding weights); 3) Script to verify the actual sparsity of the resulting model. Note that you can run them, but in my experiments I was not able to verify the resulting models.
  - `attack.py`, `data*.py`, `adv_attack.py`: code from the original ATAS repository. I believe that I do not make use of them after I referring to them when writing code.
  - `structured_prune_*.py`: the names explain themselves
- `alpha-beta-CROWN`: the formal verification tool
  - Usage: `cd` into the `complete_verifier` folder; `python abcrown.py --config your_config_file`.
  - `complete_verifier/models/`: where the actual model weights locate
  - `complete_verifier/model_defs.py`: the model architecture, need to match the ones in `Adv-train/`.
  - `complete_verifier/exp_configs/beta_crown`: the actual configurations to run the verifier. You can refer to `complete_verifier/exp_configs/tutorial_examples` to learn how to use it. I only use `./beta_crown/cifar_resnet18.yaml` and `./beta_crown/cifar_resnet_4b.yaml`.
- `convert*.py`: a small script that moves the `ResNet4b` and `ResNet18` trained in `Adv-train` to the directory of the verification tool, while ensuring the structure is compatible.

### Acknowledgement

This repository is based on the nice codebase of the following repositories:

- https://github.com/HuangZhiChao95/ATAS
- https://github.com/Verified-Intelligence/alpha-beta-CROWN

and one experiment is based on the dataset from this repository:

- https://github.com/wzekai99/DM-Improves-AT

Please only cite their works with the information provided by the original authors. You don't need to cite this repository.