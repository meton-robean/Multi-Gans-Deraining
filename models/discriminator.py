#PyTorch lib
import torch
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
from models.spectral import SpectralNorm


#Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = SpectralNorm(nn.Linear(64, 1))


    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
