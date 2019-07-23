#PyTorch lib
import torch
from models.spectral import SpectralNorm
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision

#Tools lib
import numpy as np
import cv2
import random
import time
import os
import math

def z_to_Tensor(z, tensor_size):
    list = []
    num = 1
    for i in range(len(tensor_size)):
        if(i!=1):
            num = num*tensor_size[i]
    for i in range(num):
        list.append(z)
    list = torch.Tensor(list).cuda()
    list = list.view(tensor_size[0],1, tensor_size[2], tensor_size[3])
    return list

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
#Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3+1, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU()
            )
        self.det_conv1 = ResidualBlock(64)
        self.det_conv2 = ResidualBlock(64)
        self.det_conv3 = ResidualBlock(64)
        self.det_conv4 = ResidualBlock(64)
        self.det_conv5 = ResidualBlock(64)
        self.det_conv6 = ResidualBlock(64)
        self.output = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 3, 3, 1, 1))
            )

    def forward(self, input, z):
        tensor_size = input.shape
        z_tensor = z_to_Tensor(z,tensor_size)
        x = torch.cat((input, z_tensor), 1)
        x0 = self.det_conv0(x)
        x = self.det_conv1(x0)
        x = self.det_conv2(x)
        x = self.det_conv3(x)
        x = self.det_conv4(x)
        x = self.det_conv5(x)
        x = self.det_conv6(x)
        x = self.output(x)
        return  input+x
