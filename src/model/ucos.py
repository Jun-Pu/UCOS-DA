import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import model.dino as DINO
url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
backbone = DINO.ViTFeat(url, feat_dim=768, vit_arch="base", vit_feat="k", patch_size=8)

from model.discriminator import Discriminator

class UCOSDA(nn.Module):
    def __init__(self, patch_size, train_size):
        super(UCOSDA, self).__init__()
        self.sMod = backbone
        self.flatten = nn.Unflatten(2, torch.Size([train_size // patch_size, train_size // patch_size]))
        self.tMod = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.adaMod = Discriminator(train_size // patch_size)
        self.downSample = nn.Upsample(scale_factor=1 / patch_size, mode='bicubic')
        self.out_size = train_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_c1, x_c2=None):
        if not x_c2 == None:
            # feature extraction from source domain
            with torch.no_grad():
                feats_c1 = self.sMod(x_c1)
            feats_c1 = self.flatten(feats_c1)

            with torch.no_grad():
                feats_c2 = self.sMod(x_c2)
            feats_c2 = self.flatten(feats_c2)

            # linear probe at target domain
            y_c1 = self.tMod(feats_c1)
            y_c2 = self.tMod(feats_c2)

            # group foreground&background
            fc1 = self.sigmoid(y_c1)
            bc1 = -1 * fc1 + 1
            fc2 = self.sigmoid(y_c2)
            bc2 = -1 * fc2 + 1

            # FB-aware adversarial domain adaptation
            vld_fc1 = self.adaMod(torch.cat((self.downSample(x_c1), fc1), 1))
            vld_bc1 = self.adaMod(torch.cat((self.downSample(x_c1), bc1), 1))
            vld_fc2 = self.adaMod(torch.cat((self.downSample(x_c2), fc2), 1))
            vld_bc2 = self.adaMod(torch.cat((self.downSample(x_c2), bc2), 1))

            vld_fc = (vld_fc1 + vld_fc2) / 2
            vld_bc = (vld_bc1 + vld_bc2) / 2

            # upsample to the trainsize
            y_c1 = F.interpolate(y_c1, size=self.out_size, mode='bicubic', align_corners=False)
            y_c2 = F.interpolate(y_c2, size=self.out_size, mode='bicubic', align_corners=False)

            return y_c1, vld_fc, y_c2, vld_bc
        else:
            with torch.no_grad():
                feats = self.sMod(x_c1)
            feats = self.flatten(feats)
            y = self.tMod(feats)
            y = F.interpolate(y, size=self.out_size, mode='bicubic', align_corners=False)

            return y