# VAE-Pytorch
Clone coding of [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE/tree/master) to study VAE

## 🚀 Train
```bash
python train.py -c configs/celeba.yaml
python train.py -c configs/mnist.yaml
```

## 📸 Reconstructions
| Original image                  | 2-D latent space             | 5-D latent space             | 10-D latent space             | 20-D latent space             |
|:-------------------------------:|:----------------------------:|:----------------------------:|:-----------------------------:|:-----------------------------:|
| ![](images/recons_original.png) | ![](images/recons_dim_2.png) | ![](images/recons_dim_5.png) | ![](images/recons_dim_10.png) | ![](images/recons_dim_20.png) |

## ✨ New features
- [ ] Use WandB instead of TensorBoard
- [x] Can train with MNIST dataset
- [x] Add MNIST reconstruction results
- [ ] Visualize of learned data manifold