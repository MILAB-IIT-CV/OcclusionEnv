import os
import torch
import torch.nn as nn
from skimage import img_as_ubyte
import imageio
import numpy as np

# datastructures
from pytorch3d.datasets import ShapeNetCore
from environment import OcclusionEnv
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    # Load shapenet dataset
    try:
        # From Matyi's external drive
        shapenetdir = "/Volumes/MacMiklos/M/BME/2021_12-OcclusionEnvironment/Shapenet/"
        shapenet_dataset = ShapeNetCore(shapenetdir, version=2)
    except:
        shapenet_dataset = ShapeNetCore("./data/shapenetcore", version=2)
    print("Shapenetcore dataset loaded")
    iterations = {}

    for iteration in range(105, 311, 5):     # TODO (-310, 310, 10):

        current_azimuth = iteration/3200    # [-100pi; 100pi] -> [-3,1/32; 3,1/32]

        # load shapenet environment loader
        env = OcclusionEnv(shapenet_dataset)
        obs = env.reset(azimuth=current_azimuth)

        # create folder for output
        output_path = os.path.join("./results/" + str(current_azimuth))
        try:
            os.mkdir(output_path)

        except:
            pass

        # initialize writer
        filename_output = os.path.join(output_path + "/" + str(current_azimuth) + "_optimization_demo.gif")
        writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

        action = nn.Parameter(torch.tensor([0., 0.]))
        lr = 1e-6
        rewards = []
        fullRewards = []
        print(f"Optimization start. Current azimuth value: {current_azimuth}")

        i = 0
        while True:     # TODO i < 100
            if i % 50 == 0:
                print(i)
            i += 1
            if action.grad is not None:
                action.grad.zero_()
            obs, reward, finished, info = env.step(action)
            reward.backward()
            if torch.isnan(action.grad).any():
                print("NaN gradients detected, attempting correction")
                continue
            with torch.no_grad():
                action += lr * action.grad
            env.render()
            rewards.append(reward.cpu().item())
            fullRewards.append(info['full_reward'].cpu().item())

            image = obs[0].detach().permute(1,2,0).cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)

            image = info['full_state'][0, ..., 3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)

            if finished:
                print(f"finished after {i} iteration(s).")
                iterations[current_azimuth] = i
                break

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.subplot(1, 2, 2)
        plt.plot(fullRewards)
        plt.savefig(output_path + "/" + str(current_azimuth) + "_Rewards.png")

        # writer.close()

    file = open("./results/iterations.txt", "w")
    dict_string = repr(iterations)
    file.write(dict_string)
    file.close()