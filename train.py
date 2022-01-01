from model import PredictorNet
from environment import OcclusionEnv
from SubProcVecEnv import SimpleVecEnv
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pytorch3d.datasets import ShapeNetCore

if __name__ == '__main__':

    numEnvs = 8

    # Check if shapenet is installed
    shapenetdir = "./data/shapenetcore"
    if len(os.listdir(shapenetdir)) == 0:
        envs = [lambda: OcclusionEnv() for _ in range(numEnvs)]
    else:
        shapenet_dataset = ShapeNetCore(shapenetdir, version=2)
        envs = [lambda: OcclusionEnv(shapenet_dataset) for _ in range(numEnvs)]

    env = SimpleVecEnv(envs)
    net = PredictorNet(8).cuda()

    numEpisodes = 50
    numInterations = 10

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters())
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, numEpisodes, 1e-4)

    losses = []

    for i in range(numEpisodes):

        obs = env.reset()

        running_loss = 0

        numNaN = 0

        for j in tqdm(range(numInterations)):

            step = nn.Parameter(torch.randn(numEnvs,2).cuda())
            #step.grad = torch.zeros((numEnvs,2)).cuda()
            optimizer.zero_grad()
            obs, rewards, finished, info = env.step(step)
            rewards.sum().backward()
            if torch.isnan(step.grad).any():
                numNaN += 1
                continue

            grad_pred = net(obs.detach())
            true_grad = step.grad / torch.sqrt((step.grad ** 2).sum(dim=1).unsqueeze(1) + 1e-7)
            loss = criterion(grad_pred, true_grad)
            loss.backward()
            optimizer.step()

            running_loss += loss.sum().item()

        running_loss /= (numInterations-numNaN)

        print("Episode %d finished. Loss: %.6f, NaN gradients: %d" % (i+1, running_loss, numNaN))

        scheduler.step()

        losses.append(running_loss)

    plt.plot(losses)
