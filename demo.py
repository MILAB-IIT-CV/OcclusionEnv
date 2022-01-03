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

    # Teapot object loader start
    """
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj("./data/teapot.obj")
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    verts2 = verts + torch.tensor([0, 0, 2.0])

    verts3 = torch.vstack([verts, verts2])
    faces3 = torch.vstack([faces, faces + verts.shape[0]])
    # Initialize each vertex to be white in color.
    verts_rgb3 = torch.ones_like(verts3)[None]  # (1, V, 3)
    textures3 = TexturesVertex(verts_features=verts_rgb3.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    teapot2_mesh = Meshes(
        verts=[verts2.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    full_mesh = Meshes(
        verts=[verts3.to(device)],
        faces=[faces3.to(device)],
        textures=textures3
    )
    
    env = OcclusionEnv()
    obs = env.reset(meshes=[full_mesh, teapot_mesh, teapot2_mesh])
    # teapot loader object end
    """

    # shapenet environment loader from here
    env = OcclusionEnv(shapenet_dataset)
    #env = OcclusionEnv()
    print("class instantiated")
    obs = env.reset(azimuth=np.random.default_rng().uniform(low=-40, high=40))
    print("env reset complete")
    # shapenet environment loader until here

    action = nn.Parameter(torch.tensor([0.,0.]))

    filename_output = "./teapot_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
    filename_output = "./teapot_optimization_demo_occl.gif"
    writer2 = imageio.get_writer(filename_output, mode='I', duration=0.3)

    lr = 1e-6

    rewards = []
    fullRewards = []
    print("optimization start")
    i = 0
    while i < 100:
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
        writer2.append_data(image)

        if finished:
            break

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.subplot(1, 2, 2)
    plt.plot(fullRewards)
    plt.show()
    plt.savefig("Rewards.png")

    writer.close()
    writer2.close()