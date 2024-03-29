import cv2
import os
import torch
import torch.nn as nn
from skimage import img_as_ubyte
import imageio
import numpy as np

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.datasets import ShapeNetCore
from environment import OcclusionEnv
import matplotlib.pyplot as plt

useShapeNet = True

if __name__ == '__main__':

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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

    azimuth_randn = np.pi / 32  # 5,625 deg
    azimuth_randn = np.pi/4  # 90 deg
    #azimuth_randn = 0  # 0 deg

    # azimuth_randn np.random.default_rng().uniform(low=-np.pi/32, high=np.pi/32)

    # Shapenet environment loader from here
    env = OcclusionEnv(shapenet_dataset)
    env.renderMode = 'human'
    print("class instantiated")
    obs = env.reset(azimuth=azimuth_randn)
    print(f"env.azimuth is {env.azimuth}")
    print("env reset complete")
    # Shapenet environment loader until here

    action = nn.Parameter(torch.tensor([0., 0.]))

    filename_output = "./optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
    filename_output = "./optimization_demo_occl.gif"
    writer2 = imageio.get_writer(filename_output, mode='I', duration=0.3)
    filename_output = "./optimization_demo_depth.gif"
    writer3 = imageio.get_writer(filename_output, mode='I', duration=0.3)

    lr = 1e-4

    rewards = []
    fullRewards = []
    print("optimization start")
    i = 0
    while True:
        # print(i)
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
        #env.render()
        rewards.append(reward.cpu().item())
        fullRewards.append(info['full_reward'].cpu().item())

        image_arr = obs[0].detach().permute(1, 2, 0).cpu().numpy()
        image = img_as_ubyte(image_arr[...,:3])
        writer.append_data(image)

        image = info['full_state'][0, ..., 3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer2.append_data(image)

        depth = image_arr[...,3]
        depth[depth == -1] = 0
        depth *= 51
        writer3.append_data(depth.astype('uint8'))


        print(fullRewards[-1])

        if finished:
            print(f"finished after {i} iteration(s).")
            break

    '''plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.subplot(1, 2, 2)
    plt.plot(fullRewards)
    plt.show()
    plt.savefig("Rewards.png")'''

    writer.close()
    writer2.close()
