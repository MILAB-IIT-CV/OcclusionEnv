import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

from model import Model, Robot

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

if __name__ == '__main__':

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Select the viewpoint using spherical angles
    distance = 4  # distance from camera to the object
    elevation = 40.0  # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    # Render the teapot providing the values of R and T.
    silhouette = silhouette_renderer(meshes_world=full_mesh, R=R, T=T)
    image_ref = phong_renderer(meshes_world=full_mesh, R=R, T=T)

    silhouette = silhouette.cpu().numpy()
    image_ref = image_ref.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)

    # We will save images periodically and compose them into a GIF.
    filename_output = "./teapot_optimization_demo.gif"
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    # Initialize a model using the renderer, mesh and reference image
    # model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)
    model = Robot(meshes=[teapot_mesh, teapot2_mesh], renderer=silhouette_renderer).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    plt.figure(figsize=(10, 10))

    ''' _, image_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
    plt.grid(False)
    plt.title("Starting position")

    plt.subplot(1, 2, 2)
    plt.imshow(model.image_ref.cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference silhouette")'''

    limits = torch.tensor([
        [-np.pi / 2, np.pi / 2],
        [-np.pi / 4, np.pi / 4], ]).to(device=model.device)

    loop = tqdm(range(200))
    for i in loop:
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        if torch.isnan(model.pos_params.grad).any():
            print("NaN gradients detected, attempting correction")
            with torch.no_grad():
                model.pos_params += torch.randn_like(model.pos_params) * 0.01
            continue
        optimizer.step()

        with torch.no_grad():
            torch.clip(model.pos_params[0], min=limits[0, 0], max=limits[0, 1])
            torch.clip(model.pos_params[1], min=limits[1, 0], max=limits[1, 1])

        loop.set_description('Optimizing (loss %.4f)' % loss.data)

        if loss.item() <= 0:
            break

        # Save outputs to create a GIF.
        if True:
            R = look_at_rotation(model.camera_position[None, :], device=model.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)
            image = phong_renderer(meshes_world=full_mesh.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)

            plt.figure()
            plt.imshow(image[..., :3])
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.axis("off")

    writer.close()
