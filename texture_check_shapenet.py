import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
    HardFlatShader,
)

from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    SHAPENET_PATH = "data/shapenetcore"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

    print("DEBUG: dataset load complete")

    shapenet_model = shapenet_dataset[0]
    print("This model belongs to the category " + shapenet_model["synset_id"] + ".")
    print("This model has model id " + shapenet_model["model_id"] + ".")
    model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]

    print("DEBUG: shapenet model loaded")

    model_textures = TexturesVertex(verts_features=torch.ones_like(model_verts, device=device)[None])
    shapenet_model_mesh = Meshes(
        verts=[model_verts.to(device)],
        faces=[model_faces.to(device)],
        textures=model_textures
    )

    # Rendering settings.
    R, T = look_at_view_transform(1.0, 1.0, 270)
    cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=512, cull_backfaces=True)
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None], device=device)


    images_by_model_ids = shapenet_dataset.render(
        model_ids=[
            "2d7562f5bf2c7f2da1d85548168d6015",
        ],
        device=device,
        cameras=cameras,
        raster_settings=raster_settings,
        lights=lights,
        shader_type=HardFlatShader
    )

    print('DEBUG: render complete')

    plt.figure(figsize=(10, 10))
    plt.imshow(images_by_model_ids[0, ..., :3].cpu().numpy())
    plt.axis("off");

    plt.show()