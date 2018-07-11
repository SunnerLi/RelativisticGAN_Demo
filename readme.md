# RelativisticGAN Demo

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

Abstraction
---
This repository simply demonstrates to generate the MNIST digit data with relativistic idea. Most important, we provides [the compatible version of loss script](https://github.com/SunnerLi/RelativiticGAN_Demo/blob/master/loss.py). You can just substitute with the relativistic version. However, the relativistic loss will not work until you revise the optimization part in your own code.     

Requirement
---
1. OpenCV
2. Torchvision
3. [Torchvision_sunner](https://github.com/SunnerLi/Torchvision_sunner)

Usage
---
We provides traditional GAN `SGAN`, GAN with relativistic idea `RSGAN`, average version of RSGAN `RaSGAN` and relativistic version of LSGAN `RaLSGAN`. You can just type the command:
```
# Train the traditional GAN
$ python3 train.py --type SGAN --epoch 100 --det SGAN

# Train for whole type of relativistic version model:
# python3 train.py --type SGAN --epoch 100 --det SGAN && python3 train.py --type RSGAN --epoch 100 --det RSGAN && python3 train.py --type RaSGAN --epoch 100 --det RaSGAN && python3 train.py --type RaLSGAN --epoch 100 --det RaLSGAN
```

TODO
---
- [ ] Add the training result for 100 epoch
- [ ] Add the training loss curve