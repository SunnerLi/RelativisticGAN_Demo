from model import SGAN, RSGAN, RaSGAN, RaLSGAN
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
import torchvision_sunner.transforms as sunnertransforms
import numpy as np
import argparse
import torch
import math
import cv2
import os

# Hyper-parameters
img_size = 64
sunnertransforms.quiet()

def parse():
    """
        Parse the argument

        --type      :   str object, the type of the GAN you want to use
                        This program only accept SGAN, RGAN, RaGAN and RaLSGAN currently
        --epoch     :   int object
        --det       :   The folder you want to store result in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type'      , type = str, default = 'sgan')
    parser.add_argument('--epoch'     , type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--det'       , type = str, default = './det')
    args = parser.parse_args()
    model_type = args.type.lower()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    if model_type == 'sgan':
        net = SGAN(device = device).to(device)
    elif model_type == 'rsgan':
        net = RSGAN(device = device).to(device)
    elif model_type == 'rasgan':
        net = RaSGAN(device = device).to(device)
    elif model_type == 'ralsgan':
        net = RaLSGAN(device = device).to(device)
    else:
        raise Exception("Sorry, the program doesn't support " + model_type + " GAN...")
    return args, net

def visialize(fake_img, path = None):
    """
        Visualize the render result

        Arg:    fake_img    - The generated image
                path        - The path you want to store into, default is None
    """
    img = make_grid(fake_img)
    img = sunnertransforms.asImg(img.unsqueeze(0))
    if path is None:
        cv2.imshow('show', img[0])
        cv2.waitKey(10)
    else:
        cv2.imwrite(path, img[0])

if __name__ == '__main__':
    # Parse parameter and create model
    args, net = parse()
    loss_D_list = []
    loss_G_list = []

    # Create folder, loader
    if not os.path.exists(args.det):
        os.mkdir(args.det)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('image/data', train = True, download = True, transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])),
        batch_size = args.batch_size, shuffle = True
    )

    # Train
    bar = tqdm(range(args.epoch))
    for i in bar:
        for j, (img, label) in enumerate(train_loader):
            net.optimize(img)
            info = net.getInfo()
            if j % 50 == 0:
                bar.set_description('loss_D: ' + str(info['loss_D']) + '    loss_G: ' + str(info['loss_G']))
                loss_D_list.append(math.log(info['loss_D']))
                loss_G_list.append(math.log(info['loss_G']))
                bar.refresh()
                gen_img = net.fake_img
                visialize(gen_img)
        visialize(gen_img, os.path.join(args.det, str(i) + '.png'))

    # Plot the loss curve
    plt.plot(range(len(loss_D_list)), loss_D_list, '-o', label = 'D loss')
    plt.plot(range(len(loss_G_list)), loss_G_list, '-o', label = 'G loss')
    plt.legend()
    plt.title('The loss curve (log scale)')
    plt.savefig(os.path.join(args.det, 'loss.png'))