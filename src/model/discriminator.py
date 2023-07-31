import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, msk_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4 * msk_shape ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        validity = self.model(x_flat)

        return validity