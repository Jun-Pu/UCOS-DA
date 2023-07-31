import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
from data import get_loader, test_in_train
from utils import AvgMeter, print_network, seg_loss
import cv2
import argparse
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.ucos import UCOSDA

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Epoch number')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='Training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='Training dataset size')
parser.add_argument('--patchsize', type=int, default=8, help='Patch number of the vision transformer')
parser.add_argument('--reg_weight', type=float, default=1.0, help='Weighting the regularization term')

opt = parser.parse_args()

# build models
model = UCOSDA(opt.patchsize, opt.trainsize)
print_network(model, 'UCOS-DA')
model = torch.nn.DataParallel(model)
model.to(device)

# set optimizer
segModParams, adaModParams = [], []
for name, param in model.named_parameters():
    if 'adaMod' in name:
       # print(name)
        adaModParams.append(param)
    else:
        segModParams.append(param)
optimizer = torch.optim.Adam([{'params':segModParams}, {'params':adaModParams}], opt.lr)

# set loss function
adv_loss = torch.nn.BCELoss()

# set path
salient_image_root = "/export/livia/home/vision/Yzhang/UCOS/DATA/tr/SOD/Images/"
salient_pseudo_gt_root = "/export/livia/home/vision/Yzhang/UCOS/DATA/tr/SOD/PseudoGTs/"
# salient_image_root_val = "/export/livia/home/vision/Yzhang/UCOS/DATA/val/SOD/Images/"
# salient_pseudo_gt_root_val = "/export/livia/home/vision/Yzhang/UCOS/DATA/val/SOD/PseudoGTs/"
camouflage_image_root = "/export/livia/home/vision/Yzhang/UCOS/DATA/tr/COD/Images/"
camouflage_pseudo_gt_root = "/export/livia/home/vision/Yzhang/UCOS/DATA/tr/COD/PseudoGTs/"
# camouflage_image_root_val = "/export/livia/home/vision/Yzhang/UCOS/DATA/val/COD/Images/"
# camouflage_pseudo_gt_root_val = "/export/livia/home/vision/Yzhang/UCOS/DATA/val/COD/PseudoGTs/"

save_path = 'checkpoints/'
if not os.path.exists(save_path): os.makedirs(save_path)

train_loader = get_loader(salient_image_root, salient_pseudo_gt_root,
                          camouflage_image_root, camouflage_pseudo_gt_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)
# validation_loader = test_in_train(salient_image_root_val, salient_pseudo_gt_root_val,
#                                   camouflage_image_root_val, camouflage_pseudo_gt_root_val,
#                                   valsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("UCOS-DA-Train")
logging.info("Config")
logging.info('epoch:{}; lr:{}; batchsize:{}; trainsize:{}; save_path:{}'.
             format(opt.epochs, opt.lr, opt.batchsize, opt.trainsize, save_path))

# ----------------------------------------------------------------------------------------------------------------------
best_mae = 1
best_epoch = 0

def TRAIN(train_loader, model, optimizer, epoch, save_path):
    optimizer.param_groups[0]['lr'] = opt.lr * 0.2 ** (epoch - 1)
    optimizer.param_groups[1]['lr'] = opt.lr * 0.2 ** (epoch - 1) * 0.1
    print("curret learning rate of target model: " + str(optimizer.param_groups[0]['lr']))
    print("curret learning rate of ADA module: " + str(optimizer.param_groups[1]['lr']))

    model.train()
    total_loss_record = AvgMeter()
    segC1_loss_record = AvgMeter()
    advCF_loss_record = AvgMeter()
    segC2_loss_record = AvgMeter()
    advCB_loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        # optimizer
        optimizer.zero_grad()

        # data
        imgsC1, pgtsC1, imgsC2, pgtsC2 = pack
        imgsC1, pgtsC1, imgsC2, pgtsC2 = Variable(imgsC1), Variable(pgtsC1), Variable(imgsC2), Variable(pgtsC2)
        imgsC1, pgtsC1, imgsC2, pgtsC2 = imgsC1.to(device), pgtsC1.to(device), imgsC2.to(device), pgtsC2.to(device)
        advGTsCF, advGTsCB = torch.zeros(opt.batchsize, 1), torch.ones(opt.batchsize, 1)
        advGTsCF.requires_grad = False
        advGTsCB.requires_grad = False
        advGTsCF, advGTsCB = advGTsCF.to(device), advGTsCB.to(device)

        # forward
        preds_c1, prob_cf, preds_c2, prob_cb = model(imgsC1, imgsC2)
        segC1_loss = seg_loss(preds_c1, pgtsC1)
        advCF_loss = adv_loss(prob_cf, advGTsCF)
        segC2_loss = seg_loss(preds_c2, pgtsC2)
        advCB_loss = adv_loss(prob_cb, advGTsCB)
        total_loss = segC1_loss + opt.reg_weight * advCF_loss + segC2_loss + opt.reg_weight * advCB_loss

        # back-propagation
        total_loss.backward()
        optimizer.step()

        total_loss_record.update(total_loss.data, opt.batchsize)
        segC1_loss_record.update(segC1_loss.data, opt.batchsize)
        advCF_loss_record.update(advCF_loss.data, opt.batchsize)
        segC2_loss_record.update(segC2_loss.data, opt.batchsize)
        advCB_loss_record.update(advCB_loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                  'Seg Cls1 Loss: {:.4f}, Adv ClsF Loss: {:.4f}, Seg Cls2 Loss: {:.4f}, Adv ClsB Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epochs, i, total_step, total_loss_record.show(),
                         segC1_loss_record.show(), advCF_loss_record.show(),
                         segC2_loss_record.show(), advCB_loss_record.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                  'Seg Cls1 Loss: {:.4f}, Adv ClsF Loss: {:.4f}, Seg Cls2 Loss: {:.4f}, Adv ClsB Loss: {:.4f}'.
                         format(epoch, opt.epochs, i, total_step, total_loss_record.show(),
                         segC1_loss_record.show(), advCF_loss_record.show(),
                                segC2_loss_record.show(), advCB_loss_record.show()))

    if epoch % 1 == 0:
        torch.save(model.state_dict(), save_path + 'UCOS-DA' + '_%d' % epoch + '.pth')

def TEST(validation_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum_c1, mae_sum_c2 = 0, 0
        for i in range(validation_loader.size):
            image_c1, pgt_c1, name_c1, HH_c1, WW_c1, \
                image_c2, pgt_c2, name_c2, HH_c2, WW_c2 = validation_loader.load_data()

            pgt_c1 = np.asarray(pgt_c1, np.float32)
            pgt_c1 /= (pgt_c1.max() + 1e-8)
            image_c1 = image_c1.to(device)

            pgt_c2 = np.asarray(pgt_c2, np.float32)
            pgt_c2 /= (pgt_c2.max() + 1e-8)
            image_c2 = image_c2.to(device)

            res_c1, _, res_c2, _ = model(image_c1, image_c2)

            res_c1 = F.upsample(res_c1, size=[WW_c1, HH_c1], mode='bilinear', align_corners=False)
            res_c1 = res_c1.sigmoid().data.cpu().numpy().squeeze()
            mae_sum_c1 += np.sum(np.abs(res_c1 - pgt_c1)) * 1.0 / (pgt_c1.shape[0] * pgt_c1.shape[1])

            res_c2 = F.upsample(res_c2, size=[WW_c2, HH_c2], mode='bilinear', align_corners=False)
            res_c2 = res_c2.sigmoid().data.cpu().numpy().squeeze()
            mae_sum_c2 += np.sum(np.abs(res_c2 - pgt_c2)) * 1.0 / (pgt_c2.shape[0] * pgt_c2.shape[1])

        mae = (mae_sum_c1 + mae_sum_c2) / (2 * validation_loader.size)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'UCOS-DA_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Let's go!")
    for epoch in range(1, (opt.epochs+1)):
        TRAIN(train_loader, model, optimizer, epoch, save_path)
        #TEST(validation_loader, model, epoch, save_path)
    print("Training Done!")