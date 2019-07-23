from models import discriminator
from models import generator
import torch
from torch import nn

class LPAGAN(nn.Module):
    def __init__(self, n_level):
        super(LPAGAN, self).__init__()
        self.n_level = n_level
        self.Generator = []
        self.Discriminator = []
        for i in range(n_level):
            g=generator.Generator()
            g.cuda()
            d=discriminator.Discriminator()
            d.cuda()
            self.Generator.append(g)
            self.Discriminator.append(d)



