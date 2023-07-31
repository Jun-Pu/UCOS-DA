import os
import time

import numpy as np
import scipy.ndimage as ndimage
import torch
from torchvision import transforms


class Eval_thread():
    def __init__(self, loader, method, dataset, output_dir, cuda):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.logfile = os.path.join(output_dir, 'results.txt')

    def run(self):
        start_time = time.time()

        # mean absolute error
        mae = self.Eval_mae()

        # f-measure & precision & recall
        fm, prec, recall = self.Eval_f_measure()
        max_fm = fm.max().item()
        mean_fm = fm.mean().item()

        log_fm = open(os.getcwd() + '/' + self.dataset + '_' + self.method + '_FMeasure' + '.txt', 'w')
        for idx_fm in range(255):
            log_fm.write(str(fm[idx_fm].item()) + '\n')
        log_fm.close()

        # auc
        #auc, TPR, FPR = self.Eval_auc()

        # e-measure
        em = self.Eval_e_measure()
        max_em = em.max().item()
        mean_em = em.mean().item()

        log_em = open(os.getcwd() + '/' + self.dataset + '_' + self.method + '_EMeasure' + '.txt', 'w')
        for idx_em in range(255):
            log_em.write(str(em[idx_em].item()) + '\n')
        log_em.close()

        # s-measure
        sm = self.Eval_s_measure()

        # mean intersection over union
        mean_iou = self.Eval_mIoU()

        # accuracy
        acc = self.Eval_accuracy()

        # weighted f-measure
        wfm = self.Eval_wf_measure()

        # logs
        self.LOG(
            '{} ({}): {:.4f} mIoU || {:.4f} Accuracy || {:.4f} max F-measure || {:.4f} mean F-measure '
            '|| {:.4f} weighted F-measure || {:.4f} S-measure || {:.4f} max E-measure || {:.4f} mean E-measure '
            '|| {:.4f} MAE.\n'
            .format(self.dataset, self.method, mean_iou, acc, max_fm, mean_fm, wfm, sm, max_em, mean_em, mae))

        return '[cost:{:.4f}s] {} ({}): {:.4f} mIoU || {:.4f} Accuracy || {:.4f} max F-measure || ' \
               '{:.4f} mean F-measure || {:.4f} weighted F-measure || {:.4f} S-measure || {:.4f} max E-measure ' \
               ' {:.4f} mean E-measure || {:.4f} MAE.'\
            .format(time.time() - start_time, self.dataset, self.method, mean_iou, acc, max_fm, mean_fm, wfm, sm,
                    max_em, mean_em, mae)

    def Eval_mIoU(self):
        print('eval[mIoU]:{} dataset with {} method.'.format(self.dataset, self.method))

        mean_IoU, n_img = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                pred, gt = (pred > 0.5).to(torch.bool), (gt > 0.5).to(torch.bool)
                intersection = torch.sum(pred * (pred == gt), dim=[-1, -2]).squeeze()
                union = torch.sum(pred + gt, dim=[-1, -2]).squeeze()
                iou = (intersection.to(torch.float) / union).mean().item()
                if iou == iou:  # for NAN
                    mean_IoU += iou
                    n_img += 1.0
            mean_IoU /= n_img

            return mean_IoU

    def Eval_accuracy(self):
        print('eval[Accuracy]:{} dataset with {} method.'.format(self.dataset, self.method))

        accuracy, n_img = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                pred, gt = (pred > 0.5).to(torch.bool), (gt > 0.5).to(torch.bool)
                acc = torch.mean((pred == gt).to(torch.float)).item()
                if acc == acc:  # for NAN
                    accuracy += acc
                    n_img += 1.0
            accuracy /= n_img

            return accuracy

    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))

        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num

            return avg_mae.item()

    def Eval_f_measure(self):
        print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))

        beta2 = 0.3
        avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            Fm = avg_f / img_num
            avg_p = avg_p / img_num
            avg_r = avg_r / img_num

            return Fm, avg_p, avg_r

    def Eval_wf_measure(self):
        print('eval[W-FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))

        threshold_sal, upper_sal, lower_sal = 0.5, 1, 0
        beta2 = 1.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            scores = 0
            imgs_num = 0
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                pred = pred.detach().cpu().numpy()[0]
                gt = gt.detach().cpu().numpy()[0]

                #if np.mean(gt) == 0:  # the ground truth is totally black
                #    scores += 1 - np.mean(pred)
                #    imgs_num += 1
                #else:
                if not np.all(np.isclose(gt, 0) | np.isclose(gt, 1)):
                    gt[gt > threshold_sal] = upper_sal
                    gt[gt <= threshold_sal] = lower_sal
                    # raise ValueError("'gt' must be a 0/1 or boolean array")
                gt_mask = np.isclose(gt, 1)
                not_gt_mask = np.logical_not(gt_mask)

                E = np.abs(pred - gt)
                dist, idx = ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

                # Pixel dependency
                Et = np.array(E)
                # To deal correctly with the edges of the foreground region:
                Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
                sigma = 7.0
                EA = ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma, mode='constant', cval=0.0)
                min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

                # Pixel importance
                B = np.ones(gt.shape)
                B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
                Ew = min_E_EA * B

                # Final metric computation
                eps = np.spacing(1)
                TPw = np.sum(gt) - np.sum(Ew[gt_mask])
                FPw = np.sum(Ew[not_gt_mask])
                R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
                P = TPw / (eps + TPw + FPw)  # Weighted Precision

                # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
                Q = (1 + beta2) * (R * P) / (eps + R + (beta2 * P))
                scores += Q
                imgs_num += 1

            return scores / imgs_num

    def Eval_auc(self):
        print('eval[AUC]:{} dataset with {} method.'.format(self.dataset, self.method))

        avg_tpr, avg_fpr, avg_auc, img_num = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                TPR, FPR = self._eval_roc(pred, gt, 255)
                avg_tpr += TPR
                avg_fpr += FPR
                img_num += 1.0
            avg_tpr = avg_tpr / img_num
            avg_fpr = avg_fpr / img_num

            sorted_idxes = torch.argsort(avg_fpr)
            avg_tpr = avg_tpr[sorted_idxes]
            avg_fpr = avg_fpr[sorted_idxes]
            avg_auc = torch.trapz(avg_tpr, avg_fpr)

            return avg_auc.item(), avg_tpr, avg_fpr

    def Eval_e_measure(self):
        print('eval[EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))

        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            Em = torch.zeros(255)
            if self.cuda:
                Em = Em.cuda()
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                Em += self._eval_e(pred, gt, 255)
                img_num += 1.0

            Em /= img_num

            return Em

    def Eval_s_measure(self):
        print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))

        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    Q = alpha * self._S_object(
                        pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num

            return avg_q

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            if torch.mean(y) == 0.0:  # the ground-truth is totally black
                y_pred_th = torch.mul(y_pred_th, -1)
                enhanced = torch.add(y_pred_th, 1)
            elif torch.mean(y) == 1.0:  # the ground-truth is totally white
                enhanced = y_pred_th
            else:  # normal cases
                fm = y_pred_th - y_pred_th.mean()
                gt = y - y.mean()
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

        return score

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                    1e-20)
        return prec, recall

    def _eval_roc(self, y_pred, y, num):
        if self.cuda:
            TPR, FPR = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            fp = (y_temp * (1 - y)).sum()
            tn = ((1 - y_temp) * (1 - y)).sum()
            fn = ((1 - y_temp) * y).sum()

            TPR[i] = tp / (tp + fn + 1e-20)
            FPR[i] = fp / (fp + tn + 1e-20)

        return TPR, FPR

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)

        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0

        return Q

    def Eval_AP(self, prec, recall):
        # Ref:
        # https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
        print('eval[AP]:{} dataset with {} method.'.format(
            self.dataset, self.method))
        ap_r = np.concatenate(([0.], recall, [1.]))
        ap_p = np.concatenate(([0.], prec, [0.]))
        sorted_idxes = np.argsort(ap_r)
        ap_r = ap_r[sorted_idxes]
        ap_p = ap_p[sorted_idxes]
        count = ap_r.shape[0]

        for i in range(count - 1, 0, -1):
            ap_p[i - 1] = max(ap_p[i], ap_p[i - 1])

        i = np.where(ap_r[1:] != ap_r[:-1])[0]
        ap = np.sum((ap_r[i + 1] - ap_r[i]) * ap_p[i + 1])

        return ap
