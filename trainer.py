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
    def __init__(self, descr="", lr=1e-3, decay=1e-5, entropy=1e-4):

        self.descr = descr if descr is not None else ""
        self.useDice = True
        self.residual = True
        self.separable = True
        self.usel1 = True
        self.dilation = 2
        self.lr = lr
        self.decay = decay
        self.entropy = entropy


        self.updateDescription()

        self.cores = os.cpu_count()

        self.makeDirs()

        self.numEpisodes = 5000

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.createEnv()
        self.createModel()
        self.createLearners()
        self.createAccumulators()

    def makeDirs(self):
        os.makedirs("./RLmodels", exist_ok=True)
        os.makedirs("./RLfigs", exist_ok=True)

    def updateDescription(self):
        if self.entropy:
            self.descr = self.descr + "_entr"

    def createEnv(self):

        shapenet_dataset = None
        if useShapeNet:
            # Load shapenet dataset
            try:
                # From Matyi's external drive
                shapenetdir = "/Volumes/MacMiklos/M/BME/2021_12-OcclusionEnvironment/Shapenet/"
                shapenet_dataset = ShapeNetCore(shapenetdir, version=2)
            except:
                try:
                    shapenet_dataset = ShapeNetCore("./data/shapenet/shapenetcore", version=2)
                except:
                    shapenet_dataset = ShapeNetCore("/data/shapenet/shapenetcore", version=2)

            print("Shapenetcore dataset loaded")

        self.env = OcclusionEnv(shapenet_dataset)
        self.env.renderMode = 'human'
        print("class instantiated")

    def createModel(self):

        self.model = FullNetwork(8, dilation=self.dilation, residual=self.residual, separable=self.separable).cuda()

    def createLearners(self):

        self.criterion_segm = BinaryDiceLoss() if self.useDice else nn.BCELoss()
        self.criterion_grad = nn.SmoothL1Loss(beta=0.01) if self.usel1 else nn.MSELoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

    def createAccumulators(self):
        self.segm_losses = []
        self.grad_losses = []
        self.losses = []
        self.accs = []
        self.ious = []

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

            _, _, occl_pred, grad_pred = self.model(img)
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
                _, _, occl_pred, grad_pred = self.model(img)
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
