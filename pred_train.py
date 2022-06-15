from dataset import OcclusionDataset
from torch.utils.data import DataLoader
from model import PredictorNet
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time

if __name__ == '__main__':

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    batch_size = 128

    trainSet = OcclusionDataset("./Dataset/", "train")
    valSet = OcclusionDataset("./Dataset/", "val")

    trainLoader = DataLoader(trainSet,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             )
    valLoader = DataLoader(valSet,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=4,
                           )

    model = PredictorNet(8, separable=True).cuda()

    numEpochs = 30

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-6,
                            )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               numEpochs,
                                               1e-5,
                                               )

    losses = []
    accs = []
    vallosses = []
    valaccs = []
    best_loss = 1
    best_acc = 0

    for i in range(numEpochs):

        running_loss = 0
        running_acc = 0

        for data in tqdm(trainLoader):

            # Read dataset
            img, _, grad, _ = data
            img, grad = img.cuda(), grad.cuda()

            # Calculate grad angle sin and cos
            with torch.no_grad():
                alpha = torch.arctan(grad[0] / grad[1])
                alpha[torch.isnan(alpha)] = 0

            grad_alpha_sin, grad_alpha_cos = torch.sin(alpha).cuda(), torch.cos(alpha).cuda()

            grad_alpha_gt = torch.empty((len(alpha), 2), dtype=torch.float64).cuda()

            grad_alpha_gt[:, 0] = grad_alpha_sin
            grad_alpha_gt[:, 1] = grad_alpha_cos

            optimizer.zero_grad()

            grad_pred = model(img)
            loss = criterion(grad_pred.float(),
                             grad_alpha_gt.float(),
                             )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(trainLoader)

        print("Train epoch %d finished. Loss: %.6f, Accuracy: %.2f" % (i + 1,
                                                                       running_loss,
                                                                       running_acc,
                                                                       )
              )
        sys.stdout.flush()
        print(f"Train epoch {i+1}/{numEpochs} finished. Loss: {running_loss}")

        losses.append(running_loss)

        running_loss = 0

        for data in tqdm(valLoader):

            img, _, grad, _ = data
            img, grad = img.cuda(), grad.cuda()

            # Calculate grad angle sin and cos
            with torch.no_grad():
                alpha = torch.arctan(grad[0] / grad[1])
                alpha[torch.isnan(alpha)] = 0

            grad_alpha_sin, grad_alpha_cos = torch.sin(alpha).cuda(), torch.cos(alpha).cuda()

            grad_alpha_gt = torch.empty((len(alpha), 2), dtype=torch.float64).cuda()

            grad_alpha_gt[:, 0] = grad_alpha_sin
            grad_alpha_gt[:, 1] = grad_alpha_cos

            with torch.no_grad():
                grad_pred = model(img)
                loss = criterion(grad_pred.float(),
                                 grad_alpha_gt.float(),
                                 )

                running_loss += loss.item()

        running_loss /= len(valLoader)

        print(f"Val epoch {i+1}/{numEpochs} finished. Loss: {running_loss}")

        sys.stdout.flush()

        vallosses.append(running_loss)

        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model, "bestGradPredModel.pt")
            print("Best model found and saved")

        scheduler.step()

    plt.plot(losses, label="train")
    plt.plot(vallosses, label="validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig('pred_train_losses.png')
