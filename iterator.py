import os

import numpy
import numpy as np
import torch
import torch.nn as nn
from skimage import img_as_ubyte
import imageio

from tqdm import tqdm

# datastructures
from pytorch3d.datasets import ShapeNetCore
from environment import OcclusionEnv

# DL model
from model import Segmenter, FullNetwork

# visualization
import matplotlib.pyplot as plt

if __name__ == '__main__':
    export_env_gif = False
    export_pred_gif = True

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
        shapenet_dataset = ShapeNetCore("/data/shapenet/shapenetcore", version=2)

    print("Shapenetcore dataset loaded")

    # init output dictionaries
    iterations_raw = {}
    iterations_model = {}

    # create environment
    env = OcclusionEnv(shapenet_dataset)
    obs = env.reset(new_scene=True)

    # number of steps and azimuth range
    azimuth_low = -310  # [-pi/32]
    azimuth_high = 311  # [pi/32]
    azimuth_step_size = 10      # [pi/992]
    num_iters = azimuth_high - azimuth_low
    num_iters /= azimuth_step_size
    num_iters = np.round_(num_iters)

    # allowed of maximum steps per initial azimuth value
    max_step = 2000

    for iteration in tqdm(range(azimuth_low, azimuth_high, azimuth_step_size)):

        current_azimuth = iteration/3200    # [-100pi; 100pi] -> [-3,1/32; 3,1/32] -> [~-5,625deg; ~5,625deg]

        # init environment
        obs = env.reset(new_scene=False, azimuth=current_azimuth)

        # create folder for output
        output_path = os.path.join("./results/current_result/" + str(current_azimuth))
        try:
            os.mkdir(output_path)

        except:
            pass

        # initialize image writer
        if export_env_gif:
            filename_output_raw = os.path.join(output_path + "/" + str(current_azimuth) + "_gradient_raw.gif")
            writer_raw = imageio.get_writer(filename_output_raw, mode='I', duration=0.3)
        if export_pred_gif:
            filename_output_model = os.path.join(output_path + "/" + str(current_azimuth) + "_gradient_pred.gif")
            writer_model = imageio.get_writer(filename_output_model, mode="I", duration=0.3)

        # with raw gradient
        action = nn.Parameter(torch.tensor([0., 0.]))
        lr = 1e-6
        rewards = []
        fullRewards = []
        print(f"Current azimuth value: {current_azimuth}")

        # raw gradient loop
        for i in range(1500):
            print(i, end="\r")
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

            if export_env_gif:
                image = obs[0].detach().permute(1,2,0).cpu().numpy()
                image = img_as_ubyte(image[..., 2])
                writer_raw.append_data(image)

            if finished:
                print(f"Finished after {i} iteration(s) with env gradients.")
                iterations_raw[current_azimuth] = i + 1
                break
            elif i == max_step-1:
                print(f"Haven't finished in {i} steps. Continuing.")
                iterations_raw[current_azimuth] = i + 1

        # with model prediction
        obs = env.reset(new_scene=False, azimuth=current_azimuth)

        dilation = 2
        residual = True
        separable = True
        useDice = True
        usel1 = True

        # step size multiplier
        step_lr = 0.01

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
        model.train(False)

        action = nn.Parameter(torch.tensor([0., 0.]))
        rewards = []
        fullRewards = []

        # model loop
        for i in range(1500):
            print(i, end="\r")
            if action.grad is not None:
                action.grad.zero_()
            obs, reward, finished, info = env.step(action)

            env.render()

            with torch.no_grad():
                image = obs.cuda()
                _, _, _, pred_grad = model(image)

            action = - step_lr * pred_grad

            # output
            rewards.append(reward.cpu().item())
            fullRewards.append(info['full_reward'].cpu().item())

            if export_pred_gif:
                image = obs[0].detach().permute(1, 2, 0).cpu().numpy()
                image = img_as_ubyte(image[..., 2])
                writer_model.append_data(image)

            if finished:
                iterations_model[current_azimuth] = i+1
                print(f"Finished after {i} iteration(s) with predicted gradients.")
                break

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.subplot(1, 2, 2)
        plt.plot(fullRewards)
        plt.savefig(output_path + "/" + str(current_azimuth) + "_Rewards.png")

        if export_env_gif:
            writer_raw.close()
        if export_pred_gif:
            writer_model.close()

    file = open("./results/iterations_raw.txt", "w")
    dict_string = repr(iterations_raw)
    file.write(dict_string)
    file.close()

    file = open("./results/iterations_model.txt", "w")
    dict_string = repr(iterations_model)
    file.write(dict_string)
    file.close()
