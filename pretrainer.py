import pickle

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


class PreTrainer(object):
    def __init__(self, descr="", useDice=True, usel1 = False, residual=True, separable=False, dilation=1, lr=1e-3, decay=1e-5, quick_test=False):

        self.descr = descr if descr is not None else ""
        self.useDice = useDice
        self.residual = residual
        self.separable = separable
        self.usel1 = usel1
        if dilation > 2:
            dilation = 2
        if dilation < 1:
            dilation = 1
        self.dilation = dilation
        self.lr = lr
        self.decay = decay
        self.quick_test = quick_test
        self.updateDescription()

        self.cores = os.cpu_count()
        self.batch_size = 128

        self.makeDirs()

        self.numEpochs = 2 if quick_test else 50

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.createLoaders()
        self.createModel()
        self.createLearners()
        self.createAccumulators()

    def makeDirs(self):
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./figs", exist_ok=True)

    def updateDescription(self):
        if self.useDice:
            self.descr = self.descr + "_dice"
        if self.usel1:
            self.descr = self.descr + "_l1"
        if self.quick_test:
            self.descr = self.descr + "_quick_test"
        if self.dilation > 1:
            self.descr = self.descr + "_dilated"
        if self.residual:
            self.descr = self.descr + "_res"
        if self.separable:
            self.descr = self.descr + "_sep"

    def createLoaders(self):

        trainSet = OcclusionDataset("./Dataset/", "train")
        valSet = OcclusionDataset("./Dataset/", "val")

        self.trainLoader = DataLoader(trainSet, batch_size=self.batch_size, shuffle=True, num_workers=self.cores)
        self.valLoader = DataLoader(valSet, batch_size=self.batch_size, shuffle=True, num_workers=self.cores)

    def createModel(self):

        self.model = FullNetwork(8, dilation=self.dilation, residual=self.residual, separable=self.separable).cuda()

    def createLearners(self):

        self.criterion_segm = BinaryDiceLoss() if self.useDice else nn.BCELoss()
        self.criterion_grad = nn.SmoothL1Loss(beta=0.01) if self.usel1 else nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.numEpochs, self.lr * 0.1)

    def createAccumulators(self):
        self.segm_losses = []
        self.grad_losses = []
        self.losses = []
        self.accs = []
        self.ious = []
        self.val_segm_losses = []
        self.val_grad_losses = []
        self.vallosses = []
        self.valaccs = []
        self.valious = []

        self.bestIoU = 0
        self.bestAcc = 0
        self.bestLoss = 0
        self.bestSegmLoss = 0
        self.bestGradLoss = 0

    def train(self):
        running_loss = 0
        running_segm_loss = 0
        running_grad_loss = 0
        running_acc = 0
        running_iou = 0

        self.model.train()

        for data in tqdm(self.trainLoader):
            img, occlusion, grad, _ = data
            img, occlusion, grad = img.cuda(), occlusion.cuda(), grad.cuda()

            self.optimizer.zero_grad()

            _, occl_pred, grad_pred = self.model(img)
            loss_segm = self.criterion_segm(occl_pred, occlusion)
            loss_grad = self.criterion_grad(grad_pred, grad)
            loss = loss_grad + loss_segm
            loss.backward()

            pred_map = occl_pred > 0.5
            bin_occlusion = occlusion > 0.5
            pred_correct = (pred_map == bin_occlusion).sum()
            pred_acc = pred_correct / torch.numel(pred_map)

            intersection = (pred_map * bin_occlusion).sum()
            union = ((pred_map + bin_occlusion) > 0.5).sum()

            iou = intersection / union

            self.optimizer.step()

            running_loss += loss.item()
            running_segm_loss += loss_segm.item()
            running_grad_loss += loss_grad.item()
            running_acc += pred_acc.item()
            running_iou += iou.item()

        running_loss /= len(self.trainLoader)
        running_segm_loss /= len(self.trainLoader)
        running_grad_loss /= len(self.trainLoader)
        running_acc /= len(self.trainLoader) / 100.0
        running_iou /= len(self.trainLoader) / 100.0

        sys.stdout.flush()

        return running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou


    def val(self):
        running_loss = 0
        running_segm_loss = 0
        running_grad_loss = 0
        running_acc = 0
        running_iou = 0

        self.model.eval()

        for data in tqdm(self.valLoader):
            img, occlusion, grad, _ = data
            img, occlusion, grad = img.cuda(), occlusion.cuda(), grad.cuda()

            with torch.no_grad():
                _, occl_pred, grad_pred = self.model(img)
                loss_segm = self.criterion_segm(occl_pred, occlusion)
                loss_grad = self.criterion_grad(grad_pred, grad)
                loss = loss_grad + loss_segm

                pred_map = occl_pred > 0.5
                bin_occlusion = occlusion > 0.5
                pred_correct = (pred_map == bin_occlusion).sum()
                pred_acc = pred_correct / torch.numel(pred_map)

                intersection = (pred_map * bin_occlusion).sum()
                union = ((pred_map + bin_occlusion) > 0.5).sum()

                iou = intersection / union

                running_loss += loss.item()
                running_segm_loss += loss_segm.item()
                running_grad_loss += loss_grad.item()
                running_acc += pred_acc.item()
                running_iou += iou.item()

        running_loss /= len(self.valLoader)
        running_segm_loss /= len(self.valLoader)
        running_grad_loss /= len(self.valLoader)
        running_acc /= len(self.valLoader) / 100.0
        running_iou /= len(self.valLoader) / 100.0

        sys.stdout.flush()

        return running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou



    def run(self):


        print("Strating Training with options")
        print("Dice: ", self.useDice)
        print("L1: ", self.usel1)
        print("Residual: ", self.residual)
        print("Separable: ", self.separable)
        print("Dilation: ", self.dilation)
        print("LR: ", self.lr)
        print("Decay: ", self.decay)
        print("Quick Test: ", self.quick_test)
        print("")

        for epoch in range(self.numEpochs):

            running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou = self.train()

            print("Train epoch %d finished. Loss: %.6f, Dice: %.6f, MSE: %.6f, Accuracy: %.2f, IoU: %.2f" % (
                epoch + 1, running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou))
            sys.stdout.flush()
            self.losses.append(running_loss)
            self.segm_losses.append(running_segm_loss)
            self.grad_losses.append(running_grad_loss)
            self.accs.append(running_acc)
            self.ious.append(running_iou)

            running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou = self.val()

            print("Val epoch %d finished. Loss: %.6f, Dice: %.6f, MSE: %.6f, Accuracy: %.2f, IoU: %.2f" % (
                epoch + 1, running_loss, running_segm_loss, running_grad_loss, running_acc, running_iou))
            sys.stdout.flush()
            self.vallosses.append(running_loss)
            self.val_segm_losses.append(running_segm_loss)
            self.val_grad_losses.append(running_grad_loss)
            self.valaccs.append(running_acc)
            self.valious.append(running_iou)

            if running_iou > self.bestIoU:
                self.bestIoU = running_iou
                self.bestAcc = running_acc
                self.bestLoss = running_loss
                self.bestSegmLoss = running_segm_loss
                self.bestGradLoss = running_grad_loss
                torch.save(self.model, "./models/bestSegModel_" + self.descr + ".pt")
                print("Best model found")
                sys.stdout.flush()

            self.scheduler.step()

        self.save()

    def save(self):

        plt.figure()
        plt.plot(self.losses, label="train")
        plt.plot(self.vallosses, label="validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig("./figs/pred_train_losses" + self.descr + ".png")
        plt.close()

        plt.figure()
        plt.plot(self.segm_losses, label="train")
        plt.plot(self.val_segm_losses, label="validation")
        plt.xlabel('Epoch')
        plt.ylabel('Dice Loss')
        plt.legend(loc="upper right")
        plt.savefig("./figs/pred_train_losses_dice" + self.descr + ".png")
        plt.close()

        plt.figure()
        plt.plot(self.grad_losses, label="train")
        plt.plot(self.val_grad_losses, label="validation")
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend(loc="upper right")
        plt.savefig("./figs/pred_train_losses_mse" + self.descr + ".png")
        plt.close()

        data = {
            "IoU": self.bestIoU,
            "Acc": self.bestAcc,
            "Loss": self.bestLoss,
            "Dice": self.bestSegmLoss,
            "MSE": self.bestGradLoss,
        }

        with open("./figs/bestRes" + self.descr + ".pickle", "wb+") as file:
            pickle.dump(data, file)

        print("Finished Training")
        print("")
