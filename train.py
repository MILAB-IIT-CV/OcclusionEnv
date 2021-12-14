from model import PredictorNet
from environment import OcclusionEnv
from SubProcVecEnv import SubprocVecEnv
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

if __name__ == '__main__':

    numEnvs = 8

    envs = [lambda: OcclusionEnv() for _ in range(numEnvs)]
    env = SubprocVecEnv(envs)
    net = PredictorNet(8)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters())
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 1000, 1e-4)

    numEpisodes = 1000
    numInterations = 1000

    losses = []

    for i in range(numEpisodes):

        obs = env.reset()

        running_loss = 0

        for j in range(numInterations):

            step = nn.Parameter(torch.randn(2))
            optimizer.zero_grad()
            obs, reward, finished, info = env.step(step)
            grad_pred = net(obs)
            reward.backward()
            if torch.isnan(step.grad).any():
                print("NaN gradients detected, attempting correction")
                continue

            loss = criterion(grad_pred, step.grad)
            optimizer.step()

            running_loss += loss.sum().item()

        print("Episode %d finished. Reward: %.2f" % (i+1, running_loss))

        scheduler.step()

        losses.append(running_loss)

    plt.plot(losses)
