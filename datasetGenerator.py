import cv2
import os
import os.path as osp
import torch
import torch.nn as nn
from skimage import img_as_ubyte
import imageio
import numpy as np
import pickle
import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

useShapeNet = False

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
    azimuth_randn = np.pi / 2  # 90 deg

    # azimuth_randn np.random.default_rng().uniform(low=-np.pi/32, high=np.pi/32)

    # Shapenet environment loader from here
    lr = 1e-1
    env = OcclusionEnv(shapenet_dataset)
    env.renderMode = 'human'
    print("class instantiated")
    # Shapenet environment loader until here

    os.makedirs("./Dataset", exist_ok=True)

    print("optimization start")
    for i in range(2):
        # print(i)

        os.makedirs("./Dataset/run_%d/Depth" % i, exist_ok=True)
        os.makedirs("./Dataset/run_%d/RGB" % i, exist_ok=True)
        os.makedirs("./Dataset/run_%d/Occl" % i, exist_ok=True)

        posParams = np.empty((0,5))

        action = nn.Parameter(lr*torch.randn(2))
        obs = env.reset(azimuth=azimuth_randn)
        print(f"env.azimuth is {env.azimuth}")
        print("env reset complete")

        for j in tqdm.tqdm(range(10)):
            if action.grad is not None:
                action.grad.zero_()
            obs, reward, finished, info = env.step(action)
            reward.backward()
            if torch.isnan(action.grad).any():
                print("NaN gradients detected, attempting correction")
                continue

            #env.render()

            grad =  action.grad
            pos = np.array([j, env.elevation, env.azimuth, grad[0].item(), grad[1].item()])
            posParams = np.append(posParams, pos)

            image_arr = obs[0].detach().permute(1, 2, 0).cpu().numpy()
            image = img_as_ubyte(image_arr[...,:3])
            cv2.imwrite("./Dataset/run_%d/RGB/%d.jpg" % (i,j), image.astype('uint8'))

            occl = info['full_state'][0, ..., 3].detach().squeeze().cpu().numpy()
            occl_image = img_as_ubyte(occl)
            cv2.imwrite("./Dataset/run_%d/Occl/%d.png" % (i,j), occl_image.astype('uint8'))

            depth = image_arr[...,3]
            depth[depth == -1] = 0
            depth *= 51
            cv2.imwrite("./Dataset/run_%d/Depth/%d.png" % (i,j), depth.astype('uint8'))

            action = nn.Parameter(lr * torch.randn(2))


        fName = "./Dataset/run_%d/params.pickle" % i
        file = open(fName, 'wb')
        pickle.dump(posParams, file)
        print("Run %d complete" % i)
