import gym
from gym.spaces import Tuple, MultiDiscrete, Box, MultiBinary, Dict, Space, Discrete
import torch
from torch import nn as nn
import cv2
import numpy as np
import os

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj

# For debugging
import warnings
warnings.filterwarnings("ignore")


# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# ShapeNet components
from pytorch3d.datasets import (
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from torch.utils.data import DataLoader


def load_shapenet_meshes(dataset):
    # Set "mute" to True, if no printing is necessary
    mute = False

    # Distance is the displacement of the farther object
    distance = 2

    # Choose an object category randomly, then choose model id for that category randomly
    object_indices = []
    for _ in range(2):
        category_randn = np.random.default_rng().integers(low=len(dataset.synset_dict))
        category_name = list(dataset.synset_dict.keys())[category_randn]

        low_idx = dataset.synset_start_idxs[category_name]
        high_idx = low_idx + dataset.synset_num_models[category_name]

        object_indices.append(np.random.default_rng().integers(low=low_idx, high=high_idx))

    # initialize obj#1
    obj_1 = dataset[object_indices[0]]
    obj_1_verts, obj_1_faces = obj_1["verts"], obj_1["faces"]

    if not mute:
        print(f"object 1 index is: {object_indices[0]}.")
        print(obj_1["synset_id"])
        print("Model 1 belongs to the category " + obj_1["label"] + ".")
        print("Model 1 has model id " + obj_1["model_id"] + ".")

    # white vertices
    obj_1_textures = TexturesVertex(verts_features=torch.ones_like(obj_1_verts, device=device)[None])
    obj_1_mesh = Meshes(
        verts=[obj_1_verts.to(device)],
        faces=[obj_1_faces.to(device)],
        textures=obj_1_textures
    )

    # initialize obj#2
    obj_2 = dataset[object_indices[1]]
    obj_2_verts, obj_2_faces = obj_2["verts"] + torch.tensor([0, 0, distance]), obj_2["faces"]

    if not mute:
        print(f"object 2 index is: {object_indices[1]}.")
        print(obj_2["synset_id"])
        print("Model 2 belongs to the category " + obj_2["label"] + ".")
        print("Model 2 has model id " + obj_2["model_id"] + ".")

    # white vertices
    obj_2_textures = TexturesVertex(verts_features=torch.ones_like(obj_2_verts, device=device)[None])
    obj_2_mesh = Meshes(
        verts=[obj_2_verts.to(device)],
        faces=[obj_2_faces.to(device)],
        textures=obj_2_textures
    )

    verts3 = torch.vstack([obj_1_verts, obj_2_verts])
    faces3 = torch.vstack([obj_1_faces, obj_2_faces + obj_1_verts.shape[0]])

    # Initialize each vertex to be white in color.
    textures3 = TexturesVertex(verts_features=torch.ones_like(verts3, device=device)[None])

    full_mesh = Meshes(
        verts=[verts3.to(device)],
        faces=[faces3.to(device)],
        textures=textures3
    )

    return [full_mesh, obj_1_mesh, obj_2_mesh]


def load_default_meshes():
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

    return [full_mesh, teapot_mesh, teapot2_mesh]


class OcclusionEnv():
    def __init__(self, data=None, img_size=256):
        super().__init__()

        self.metadata = "Blablabla"
        # :D

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Shapenet dataset to be passed as "data" when calling the constructor.
        self.shapenet_dataset = data

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
        self.renderMode = ""  # 'human'

    def reset(self, radius=4.0, azimuth=0.0, elevation=0.0):

        # Check if constructor is called with shapenet dataset, if not, call default (teapot) object mesh loader
        if self.shapenet_dataset is None:
            self.meshes = load_default_meshes()
        else:
            self.meshes = load_shapenet_meshes(dataset=self.shapenet_dataset)

        self.fullReward = 0
        self.camera_position = torch.zeros(3).to(self.device)

        self.radius = torch.tensor([radius]).float().to(self.device)
        self.elevation = torch.tensor([elevation]).float().to(self.device)  # angle of elevation in degrees
        self.azimuth = torch.tensor([azimuth]).float().to(self.device)

        R, T = look_at_view_transform(self.radius, self.elevation, self.azimuth, device=self.device)

        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T).permute(0, 3, 1, 2)

        return observation

    def reset_default(self, meshes=load_default_meshes(), radius=4.0, azimuth=None, elevation=0.0):

        self.fullReward = 0

        # Set cameare x and y position randomly sampled from [-1 1] uniform distribution & on the z = 0 plane
        camera_disp_x = np.random.uniform(low=-1, high=1)
        camera_disp_y = np.random.uniform(low=-1, high=1)
        self.camera_position = torch.tensor([camera_disp_x, camera_disp_y, 0]).to(self.device)

        self.radius = torch.tensor([radius]).float().to(self.device)
        self.elevation = torch.tensor([elevation]).float().to(self.device)  # angle of elevation in degrees

        #Initialize camera azimuth randomly from -0.5 to 0.5 radian
        if azimuth is None:
            azimuth_random = np.random.uniform(low=-0.5, high=0.5)
            self.azimuth = torch.tensor([azimuth_random]).float().to(
                self.device)
        else:
            self.azimuth = torch.tensor([azimuth]).float().to(
                self.device)

        self.meshes = meshes
        R, T = look_at_view_transform(self.radius, self.elevation, self.azimuth, device=self.device)

        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T).permute(0, 3, 1, 2)

        return observation

    def render(self):

        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)
        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T)

        if self.renderMode == 'human':
            obs_img = (observation.detach().squeeze().cpu().numpy()[..., :3] * 255).astype('uint8')
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

        observation = self.phong_renderer(meshes_world=self.meshes[0].clone(), R=R, T=T).permute(0, 3, 1, 2)

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
