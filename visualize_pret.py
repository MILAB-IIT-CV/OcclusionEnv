import pickle

import cv2

from dataset import OcclusionDataset
from torch.utils.data import DataLoader
from model import Segmenter, FullNetwork
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
from loss import BinaryDiceLoss
import os
import numpy as np
import random

if __name__ == '__main__':

    dilation = 2
    residual = True
    separable = True
    useDice = True
    usel1 = True

    os.makedirs("./visualization", exist_ok=True)

    valSet = OcclusionDataset("./Dataset/", "val")

    valLoader = DataLoader(valSet, batch_size=1, shuffle=True)

    model = FullNetwork(8, dilation=dilation, residual=residual, separable=separable).cuda()

    descr = ""
    if useDice:
        descr = descr + "_dice"
    if usel1:
        descr = descr + "_l1"
    if dilation > 1:
        descr = descr + "_dilated"
    if residual:
        descr = descr + "_res"
    if separable:
        descr = descr + "_sep"

    model = torch.load("./models/bestSegModel_" + descr + ".pt")

    model.eval()

    std = torch.tensor([0.25, 0.25, 0.25]).unsqueeze(1).unsqueeze(1)
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(1)


    for i, data in enumerate(valLoader):
        img, occlusion, grad, _ = data
        img, occlusion, grad = img.cuda(), occlusion.cuda(), grad.cuda()

        with torch.no_grad():
            _, _, occl_pred, grad_pred = model(img)

            pred_map = (occl_pred > 0.5).cpu().squeeze().numpy().astype('uint8')
            bin_occlusion = (occlusion > 0.5).cpu().squeeze().numpy().astype('uint8')

            img = img.squeeze()[:3].cpu()
            img *= std
            img += mean
            img *= 255
            img = img.permute(1,2,0).numpy().astype('uint8').copy()

            h,w = img.shape[:2]
            cx, cy = int(w//2), int(h//2)
            gtx, gty = int(cx+(20*grad[0, 0])), int(cy+(20*grad[0, 1]))
            px, py = int(cx+(20*grad_pred[0])), int(cy+(20*grad_pred[1]))

            print(img.shape)

            cv2.arrowedLine(img, (cx, cy), (gtx, gty), (255, 0, 0), 1)
            cv2.arrowedLine(img, (cx, cy), (px, py), (0, 0, 255), 1)

            occl_img = np.zeros_like(img)
            occl_img[..., 0] = bin_occlusion*255
            occl_img[..., 2] = pred_map*255

            cv2.imwrite("./visualization/occl_%d.png" % i, occl_img)
            #cv2.imwrite("./visualization/occl_pred_%d.png" % i, pred_map)
            cv2.imwrite("./visualization/image_%d.jpg" % i, img)



        if i == 31:
            break


