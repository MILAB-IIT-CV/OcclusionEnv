from model import Segmenter
from environment import OcclusionEnv
from SubProcVecEnv import SimpleVecEnv
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from pytorch3d.datasets import ShapeNetCore

if __name__=='__main__':

    numEnvs = 8

    # Check if shapenet is installed
    shapenetdir = "./data/shapenetcore"
    if not os.path.exists(shapenetdir) or len(os.listdir(shapenetdir)) == 0:
        envs = [lambda: OcclusionEnv() for _ in range(numEnvs)]
    else:
        shapenet_dataset = ShapeNetCore(shapenetdir, version=2)
        envs = [lambda: OcclusionEnv(shapenet_dataset) for _ in range(numEnvs)]

    env = SimpleVecEnv(envs)

    model = Segmenter(8).cuda()

    numEpochs = 50
    numInterations = 10

    criterion = nn.BinaryCrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, numEpochs, 1e-5)

    losses = []

    for i in range(numEpochs):

        obs = env.reset()

        running_loss = 0
        running_acc = 0
        running_iou = 0

        for j in tqdm(range(numInterations)):

            with torch.no_grad():
                step = torch.randn(numEnvs, 2).cuda()
                obs, rewards, finished, info = env.step(step)

            optimizer.zero_grad()

            occl_pred = model(obs)
            occlusion = info['full_state']
            loss = criterion(occl_pred, occlusion)
            loss.backward()

            pred_map = occl_pred > 0.5
            pred_correct = (pred_map == occlusion)
            pred_acc = pred_correct / torch.numel(pred_map)

            intersection = (pred_map*occlusion).sum()
            union = ((pred_map+occlusion) > 0).sum()

            iou = intersection/union

            optimizer.step()

            running_loss += loss.item()
            running_acc += pred_acc.item()
            running_iou += iou.item()

        running_loss /= (numInterations)
        running_acc /= (numInterations) * 100
        running_iou /= (numInterations) * 100

        print("Episode %d finished. Loss: %.6f, Accuracy: %.2f, IoU: %.2f" % (i + 1, running_loss, running_acc, running_iou))

        scheduler.step()

        losses.append(running_loss)

    plt.plot(losses)
