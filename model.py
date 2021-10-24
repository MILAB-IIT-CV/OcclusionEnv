import torch
from torch import nn as nn
import numpy as np

# rendering components
from pytorch3d.renderer import look_at_rotation

class Robot(nn.Module):
    def __init__(self, meshes, renderer):
        super().__init__()
        self.meshes = meshes
        self.device = meshes[0].device
        self.renderer = renderer

        self.pos_params = nn.Parameter(
            torch.from_numpy(np.array([0.0, 0.7], dtype=np.float32)).to(self.device))

        self.camera_position = torch.zeros(3).to(self.device)

        self.radius = torch.tensor([4.0]).float().to(self.device)

    def forward(self):
        self.camera_position = torch.zeros(3).to(self.device)

        self.camera_position[0] = self.radius * torch.sin(self.pos_params[0]) * torch.cos(self.pos_params[1])
        self.camera_position[1] = self.radius * torch.sin(self.pos_params[0]) * torch.sin(self.pos_params[1])
        self.camera_position[2] = self.radius * torch.cos(self.pos_params[0])

        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image1 = self.renderer(meshes_world=self.meshes[0].clone(), R=R, T=T)
        image2 = self.renderer(meshes_world=self.meshes[1].clone(), R=R, T=T)
        image = image1 * image2

        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3]) ** 2)
        return loss, image


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image