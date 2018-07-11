# RelativisticGAN Demo

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/image/render_result/RaLSGAN_result_50_epoch.gif)

Abstraction
---
This repository simply demonstrates to generate the MNIST digit data with relativistic idea[1]. Furthermore, we modify [the original loss definition](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py) which is adopted in official CycleGAN, and provides [the compatible version of loss script](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/loss.py). You can just substitute with the relativistic version. However, the relativistic loss will not work until you revise the optimization part in your own code. At last, the above image shows the result of LSGAN which adopting relativistic trick.      

Requirement
---
1. OpenCV
2. Torchvision
3. [Torchvision_sunner](https://github.com/SunnerLi/Torchvision_sunner)([Newest version](https://gitlab.com/SunnerLi/Torchvision_sunner))

Usage
---
We provides traditional GAN `SGAN`, GAN with relativistic idea `RSGAN`, average version of RSGAN `RaSGAN` and relativistic version of LSGAN `RaLSGAN`. You can just type the command:
```
# Train the traditional GAN
$ python3 train.py --type SGAN --epoch 100 --det SGAN

# Train for whole type of relativistic version model:
# python3 train.py --type SGAN --epoch 100 --det SGAN && python3 train.py --type RSGAN --epoch 100 --det RSGAN && python3 train.py --type RaSGAN --epoch 100 --det RaSGAN && python3 train.py --type RaLSGAN --epoch 100 --det RaLSGAN
```

Result
---
For our experiments, we train the each model for 50 epoch, recording the loss value and the render result. In this section, we shows the result of traditional GAN first (without relativistic idea). The mode collapse occurs and the loss curve is awkward at the end.    
![](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/image/render_result/SGAN_result_50_epoch.gif)
![](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/image/loss_curve/SGAN_loss_curve_50_epoch.png)
 
Next, we shows the result of GAN while using the relativistic idea. As you can see, the loss curve can converge normally. The same great converge phenomenon can be proved in LSGAN.    

![](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/image/render_result/RSGAN_result_50_epoch.gif)
![](https://github.com/SunnerLi/RelativisticGAN_Demo/blob/master/image/loss_curve/RSGAN_loss_curve_50_epoch.png)

TODO
---
- [ ] Provide the loss curve without taking log

Reference
---
[1]  Alexia Jolicoeur-Martineau, "The relativistic discriminator: a key element missing from standard GAN," arXiv: 1807.00734 [cs.LG], July 2018.