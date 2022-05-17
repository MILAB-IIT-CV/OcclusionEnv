from dataset import OcclusionDataset
from torch.utils.data import DataLoader
from model import Segmenter
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
from loss import BinaryDiceLoss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrain the NN')
    parser.add_argument('--desc', type=str, help='Description for the run (will be appended to saved model and plot')
    parser.add_argument('--dice', action='store_true', help='Description for the run (will be appended to saved model and plot')

    args = parser.parse_args()

    desc = args.desc
    usedice = args.dice

    if desc is None:
        desc = ''

    if usedice:
        desc = desc + "_dice"

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainSet = OcclusionDataset("./Dataset/", "train")
    valSet = OcclusionDataset("./Dataset/", "val")

    trainLoader = DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=8)
    valLoader = DataLoader(valSet, batch_size=128, shuffle=True, num_workers=8)

    model = Segmenter(8).cuda()

    numEpochs = 50
    
    lr = 1e-3 if usedice else 1e-4

    criterion = BinaryDiceLoss() if usedice else nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, numEpochs, lr*0.1)

    losses = []
    accs = []
    ious = []
    vallosses = []
    valaccs = []
    valious = []
    bestIoU = 0

    for i in range(numEpochs):

        running_loss = 0
        running_acc = 0
        running_iou = 0

        for data in tqdm(trainLoader):

            img, occlusion, _, _ = data
            img, occlusion = img.cuda(), occlusion.cuda()

            optimizer.zero_grad()

            _, occl_pred = model(img)
            loss = criterion(occl_pred, occlusion)
            loss.backward()

            pred_map = occl_pred > 0.5
            bin_occlusion = occlusion > 0.5
            pred_correct = (pred_map == bin_occlusion).sum()
            pred_acc = pred_correct / torch.numel(pred_map)

            intersection = (pred_map * bin_occlusion).sum()
            union = ((pred_map + bin_occlusion) > 0.5).sum()

            iou = intersection / union

            optimizer.step()

            running_loss += loss.item()
            running_acc += pred_acc.item()
            running_iou += iou.item()

        running_loss /= len(trainLoader)
        running_acc /= len(trainLoader) / 100.0
        running_iou /= len(trainLoader) / 100.0

        print("Train epoch %d finished. Loss: %.6f, Accuracy: %.2f, IoU: %.2f" % (
        i + 1, running_loss, running_acc, running_iou))
        sys.stdout.flush()
        losses.append(running_loss)
        accs.append(running_acc)
        ious.append(running_iou)

        running_loss = 0
        running_acc = 0
        running_iou = 0

        for data in tqdm(valLoader):

            img, occlusion, _, _ = data
            img, occlusion = img.cuda(), occlusion.cuda()

            with torch.no_grad():
                _, occl_pred = model(img)
                loss = criterion(occl_pred, occlusion)

                pred_map = occl_pred > 0.5
                bin_occlusion = occlusion > 0.5
                pred_correct = (pred_map == bin_occlusion).sum()
                pred_acc = pred_correct / torch.numel(pred_map)

                intersection = (pred_map * bin_occlusion).sum()
                union = ((pred_map + bin_occlusion) > 0.5).sum()

                iou = intersection / union

                running_loss += loss.item()
                running_acc += pred_acc.item()
                running_iou += iou.item()

        running_loss /= len(valLoader)
        running_acc /= len(valLoader) / 100.0
        running_iou /= len(valLoader) / 100.0

        print("Val epoch %d finished. Loss: %.6f, Accuracy: %.2f, IoU: %.2f" % (
        i + 1, running_loss, running_acc, running_iou))
        sys.stdout.flush()
        vallosses.append(running_loss)
        valaccs.append(running_acc)
        valious.append(running_iou)

        if running_iou > bestIoU:
            bestIoU = running_iou
            torch.save(model, "bestSegModel_" + desc + ".pt")
            print("Best model found")

        scheduler.step()

    plt.plot(losses)
