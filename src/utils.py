import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def seg_loss(pred, mask):
    # adaptive weighting mask
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    pred = torch.sigmoid(pred)

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + wiou).mean()

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]
        #print(c)
        #d = torch.mean(torch.stack(c))
        #print(d)
        return torch.mean(torch.stack(c))

def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


# image_root = "/export/livia/home/vision/Yzhang/DataSets/DATA_ObjSeg/COD/te/CAMO/Imgs/"
# pseudo_gt_root = "/export/livia/home/vision/Yzhang/DataSets/DATA_ObjSeg/COD/te/CAMO/GT/"
# image_root_te = "/export/livia/home/vision/Yzhang/DataSets/DATA_ObjSeg/COD/te/CAMO/Imgs/"
# pseudo_gt_root_te = "/export/livia/home/vision/Yzhang/DataSets/DATA_ObjSeg/COD/te/CAMO/GT/"
