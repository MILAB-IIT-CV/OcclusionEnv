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

if __name__=='__main__':

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

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, numEpochs, 1e-5)

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
            torch.save(model, "bestSegModel.pt")
            print("Best model found")

        scheduler.step()

    plt.plot(losses)
