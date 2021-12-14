import gym
from gym.spaces import Tuple, MultiDiscrete, Box, MultiBinary, Dict, Space, Discrete
import torch
from torch import nn as nn
import cv2
import numpy as np

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)


class OcclusionEnv():
    def __init__(self, img_size=256):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # Initialize a perspective camera.
        cameras = FoVPerspectiveCameras(device=self.device)

        # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
        # edges. Refer to blending.py for more details.
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

        self.observation_space = Box(0, 1, shape=(4, img_size, img_size))
        self.action_space = Box(low=-0.1, high=0.1, shape=(2,))
        self.renderMode = 'human'



    def reset(self, meshes, radius=4.0, azimuth=0.0, elevation=0.0):

        self.fullReward = 0

        self.camera_position = torch.zeros(3).to(self.device)

        self.radius = torch.tensor([radius]).float().to(self.device)
        self.elevation = torch.tensor([elevation]).float().to(self.device)  # angle of elevation in degrees
        self.azimuth = torch.tensor([azimuth]).float().to(self.device)  # No rotation so the camera is positioned on the +Z axis.

        self.meshes = meshes
        R, T = look_at_view_transform(self.radius, self.elevation, self.azimuth, device=self.device)

        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T)

        return observation

    def render(self):

        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T)

        if self.renderMode == 'human':
            obs_img = (observation.detach().squeeze().cpu().numpy()[..., :3]*255).astype('uint8')
            cv2.imshow("Environment", obs_img)
            cv2.waitKey(25)
        else:
            return observation

    def step(self, action):

        self.detach()

        self.elevation += action[0]
        self.azimuth += action[1]

        self.camera_position[0] = self.radius * torch.sin(self.elevation) * torch.cos(self.azimuth)
        self.camera_position[1] = self.radius * torch.sin(self.elevation) * torch.sin(self.azimuth)
        self.camera_position[2] = self.radius * torch.cos(self.elevation)

        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image1 = self.silhouette_renderer(meshes_world=self.meshes[1].clone(), R=R, T=T)
        image2 = self.silhouette_renderer(meshes_world=self.meshes[2].clone(), R=R, T=T)
        self.image = image1 * image2

        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T).permute(0,3,1,2)

        # Calculate the silhouette loss
        loss = torch.sum((self.image[..., 3]) ** 2)
        reward = self.fullReward - loss

        self.fullReward = loss.detach()

        finished = (self.fullReward == 0)

        info = {'full_state': self.image, 'position': self.camera_position, 'full_reward': self.fullReward}

        return observation, reward, finished, info

    def detach(self):
        self.elevation.detach_()
        self.azimuth.detach_()
        self.radius.detach_()
        self.camera_position.detach_()