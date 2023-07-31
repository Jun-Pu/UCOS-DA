import torch
import torch.nn.functional as F
import os, argparse

from utils import print_network
from tqdm import tqdm
import cv2
import numpy as np
import time

# ----------------------------------------------------------------------------------------------------------------------
from model.ucos import UCOSDA
from data import ObjDatasetTE

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--patchsize', type=int, default=8, help='Patch number of the vision transformer')
opt = parser.parse_args()

model = UCOSDA(opt.patchsize, opt.testsize)
print_network(model, 'UCOS-DA')

# load pretrained models
model_pretrains = torch.load("/export/livia/home/vision/Yzhang/UCOS/checkpoints/UCOS-DA_checkpoint.pth")

all_params = {}
for k, v in model.state_dict().items():
    if 'module.' + k in model_pretrains.keys():
        v = model_pretrains['module.' + k]
        all_params[k] = v

model.load_state_dict(all_params)

model.cuda()
model.eval()

dataset_path = "/export/livia/home/vision/Yzhang/DataSets/DATA_ObjSeg/COD/te/"
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

save_root = "/export/livia/home/vision/Yzhang/UCOS/outputs/"

TIME_UCOSDA = []
for dataset in test_datasets:
    save_path = os.path.join(save_root, dataset)
    if not os.path.exists(save_path): os.makedirs(save_path)
    print(dataset)
    image_root = os.path.join(dataset_path, dataset, "Imgs/")
    test_loader = ObjDatasetTE(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        # generate predictions
        time_ucosda_start = time.time()
        pred = model.forward(image)
        time_ucosda_end = time.time()
        TIME_UCOSDA.append(time_ucosda_end - time_ucosda_start)

        pred = F.upsample(pred, size=[WW, HH], mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        pred = (
            (pred.detach() > 0.5).squeeze().float()
        )

        # save predictions
        pred = 255 * pred.data.cpu().numpy()
        cv2.imwrite(os.path.join(save_path, name), pred)
print('UCOS-DA: %f FPS' % (6473 / np.sum(TIME_UCOSDA)))
print('Test Done!')
# COD FPS: (for 6473)
# SOD FPS: (for 15634)