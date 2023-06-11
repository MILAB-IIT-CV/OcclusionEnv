import datetime
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

    export_env_gif = True
    export_pred_gif = True
    export_rl_gif = True

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
        try:
            shapenet_dataset = ShapeNetCore("./data/shapenet/shapenetcore", version=2)
        except:
            shapenet_dataset = ShapeNetCore("/data/shapenet/shapenetcore", version=2)

    print("Shapenetcore dataset loaded")

    # init output dictionaries
    iterations_raw = {}
    iterations_random = {}
    iterations_model = {}
    iterations_rl = {}

    num_episodes = 10
    max_step = 50

    # create environment
    env = OcclusionEnv(shapenet_dataset)

    # directory for current results
    base_path = os.path.join("./results/test" + str(datetime.datetime.now()))
    try:
        os.makedirs(base_path, exist_ok=True)

    except:
        pass

    for iteration in tqdm(range(num_episodes)):
        obs = env.reset(new_scene=True)

        # create folder for output
        output_path = os.path.join(base_path + "/" + str(iteration))
        try:
            os.mkdir(output_path)

        except:
            pass

        # initialize image writer
        if export_env_gif:
            filename_output_raw = os.path.join(output_path + "/" + str(iteration) + "_gradient_raw.gif")
            writer_raw = imageio.get_writer(filename_output_raw, mode='I', duration=0.3)
        if export_pred_gif:
            filename_output_model = os.path.join(output_path + "/" + str(iteration) + "_gradient_pred.gif")
            writer_model = imageio.get_writer(filename_output_model, mode="I", duration=0.3)
        if export_rl_gif:
            filename_output_rl = os.path.join(output_path + "/" + str(iteration) + "_rl.gif")
            writer_rl = imageio.get_writer(filename_output_rl, mode="I", duration=0.3)

        # with raw gradient
        '''action = nn.Parameter(torch.tensor([0., 0.]))
        lr = 1
        rewards = []
        fullRewards = []

        # raw gradient loop
        for i in range(max_step):
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
                image = obs[0].detach().permute(1, 2, 0).cpu().numpy()
                image = img_as_ubyte(image[..., 2])
                writer_raw.append_data(image)

            if finished:
                print(f"Finished after {i} iteration(s) with env gradients.")
                iterations_raw[iteration] = i + 1
                break
            elif i >= max_step-1:
                print(f"Haven't finished in {i} steps. Continuing.")
                iterations_raw[iteration] = i + 1

        # with random movement
        obs = env.reset(new_scene=False)

        # with raw gradient
        action = nn.Parameter(torch.randn(2))
        lr = 1
        rewards = []
        fullRewards = []

        # raw gradient loop
        for i in range(max_step):
            print(i, end="\r")
            obs, reward, finished, info = env.step(action)
            with torch.no_grad():
                action += lr * torch.randn(2)
            env.render()
            rewards.append(reward.cpu().item())
            fullRewards.append(info['full_reward'].cpu().item())

            if finished:
                print(f"Finished after {i} iteration(s) with random movement.")
                iterations_random[iteration] = i + 1
                break
            elif i >= max_step - 1:
                print(f"Haven't finished in {i} steps. Continuing.")
                iterations_random[iteration] = i + 1

        # with model prediction
        obs = env.reset(new_scene=False)

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

        model = torch.load("./models/bestSegModel_final2" + descr + ".pt")
        model.train(False)

        action = nn.Parameter(torch.tensor([0., 0.]))
        rewards = []
        fullRewards = []

        # model loop
        for i in range(max_step):
            print(i, end="\r")
            if action.grad is not None:
                action.grad.zero_()
            obs, reward, finished, info = env.step(action)

            env.render()

            with torch.no_grad():
                image = obs.cuda()
                _, _, pred_grad = model(image)

            action = - step_lr * pred_grad

            # output
            rewards.append(reward.cpu().item())
            fullRewards.append(info['full_reward'].cpu().item())

            if export_pred_gif:
                image = obs[0].detach().permute(1, 2, 0).cpu().numpy()
                image = img_as_ubyte(image[..., 2])
                writer_model.append_data(image)

            if finished:
                iterations_model[iteration] = i+1
                print(f"Finished after {i} iteration(s) with predicted gradients.")
                break

            elif i >= max_step-1:
                print(f"Haven't finished in {i} steps. Continuing.")
                iterations_model[iteration] = i + 1'''

        model = torch.load("./PPO_preTrained/OcclusionEnv/PPO_OcclusionEnv_42_36.pth")
        model.train(False)

        action = nn.Parameter(torch.tensor([0., 0.]))
        rewards = []
        fullRewards = []

        # model loop
        for i in range(max_step):
            print(i, end="\r")

            obs, reward, finished, info = env.step(action)

            env.render()

            with torch.no_grad():
                image = obs.cuda()
                action = model.select_action(image)
                #action, _, _ = model(image)

            # output
            rewards.append(reward.cpu().item())
            fullRewards.append(info['full_reward'].cpu().item())

            if export_rl_gif:
                image = obs[0].detach().permute(1, 2, 0).cpu().numpy()
                image = img_as_ubyte(image[..., 2])
                writer_rl.append_data(image)

            if finished:
                iterations_rl[iteration] = i + 1
                print(f"Finished after {i} iteration(s) with predicted gradients.")
                break

            elif i >= max_step - 1:
                print(f"Haven't finished in {i} steps. Continuing.")
                iterations_rl[iteration] = i + 1

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.subplot(1, 2, 2)
        plt.plot(fullRewards)
        plt.savefig(output_path + "/" + str(iteration) + "_Rewards.png")

        if export_env_gif:
            writer_raw.close()
        if export_pred_gif:
            writer_model.close()

    file = open(os.path.join(base_path + "/" + "iterations_raw_3.txt"), "w")
    dict_string = repr(iterations_raw)
    file.write(dict_string)
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_random_3.txt"), "w")
    dict_string = repr(iterations_random)
    file.write(dict_string)
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_model_3.txt"), "w")
    dict_string = repr(iterations_model)
    file.write(dict_string)
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_rl_3.txt"), "w")
    dict_string = repr(iterations_rl)
    file.write(dict_string)
    file.close()
