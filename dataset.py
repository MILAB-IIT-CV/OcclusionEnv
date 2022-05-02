import torch
from torchvision import transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import os
import os.path as osp
import fnmatch
import glob
import pickle

class OcclusionDataset(Dataset):
    def __init__(self, root, split="", size=(256,256)):
        super().__init__()

        self.root = root
        self.split = split

        self.size = size

        self.images = []
        self.depthImages = []
        self.labelImages = []
        self.posLabels = []
        self.gradLabels = []

        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)
        self.norm = transforms.Normalize(
            [0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25]
        )

        baseDir = osp.join(root, split)

        self.images = sorted(glob.glob(baseDir + "/**/RGB/*.jpg", recursive=True))
        self.depthImages = sorted(glob.glob(baseDir + "/**/Depth/*.png", recursive=True))
        self.labelImages = sorted(glob.glob(baseDir + "/**/Occl/*.png", recursive=True))
        labelFiles = sorted(glob.glob(baseDir + "/**/*.pickle", recursive=True))

        for lFile in labelFiles:
            file = open(lFile, "rb")
            arr = pickle.load(file)
            arr = arr.reshape([-1, 5])
            self.posLabels.append(arr[:,1:3])
            self.gradLabels.append(arr[:, 3:5])

        self.posLabels = np.concatenate(self.posLabels, 0)
        self.gradLabels = np.concatenate(self.gradLabels, 0)

    def __len__(self):
        return self.posLabels.shape[0]

    def __getitem__(self, i):
        imgfName = self.images[i]
        depthfName = self.depthImages[i]
        labelfName = self.labelImages[i]

        img = Image.open(imgfName).convert("RGB")
        depth = Image.open(depthfName).convert("I")
        label = Image.open(labelfName).convert("1")

        pos = torch.tensor(self.posLabels[i])
        grad = torch.tensor(self.gradLabels[i])

        img = img.resize(self.size)
        depth = depth.resize(self.size)
        label = label.resize(self.size)

        if self.split == "train":
            img = self.jitter(img)

            if torch.rand(1) > 0.5:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(method=Image.FLIP_LEFT_RIGHT)
                label = label.transpose(method=Image.FLIP_LEFT_RIGHT)

        img = torch.tensor(np.asarray(img))/255.0
        depth = torch.tensor(np.asarray(depth)).unsqueeze(2)/255.0

        img = torch.cat([img, depth], 2).permute(2,0,1)
        img = self.norm(img)
        label = torch.tensor(np.asarray(label)).float().unsqueeze(0)/255.0

        return img, label, grad, pos
