import torch
from torch import nn as nn
import numpy as np

# rendering components
#from pytorch3d.renderer import look_at_rotation

class Conv(nn.Module):
    def __init__(self, inch, ch, k_size, stride, dilation, bias, separable):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inch, inch, (k_size, 1), stride=(1, 1), dilation=(dilation, 1), padding=((k_size + dilation - 1) // 2, 0),
                      bias=False, groups=inch),
            nn.Conv2d(inch, inch, (1, k_size), stride=(1, 1), dilation=(1, dilation), padding=(0, (k_size + dilation - 1) // 2),
                      bias=False, groups=inch),
            nn.Conv2d(inch, ch, 1, stride=1, bias=bias),
        ) if separable else \
            nn.Conv2d(inch, ch, k_size, stride=stride, dilation=dilation, padding=(k_size+dilation-1)//2, bias=bias)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))

class TrConv(nn.Module):
    def __init__(self, inch, ch, k_size, stride, dilation, bias):
        super().__init__()

        self.conv = nn.ConvTranspose2d(inch, ch, k_size, stride=stride, dilation=dilation, padding=(k_size+dilation-1)//2, bias=bias, output_padding=1)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, ch, k_size, numLayers, dilation, bias, residual=False, separable=False):
        super().__init__()

        self.net = nn.Sequential()
        for i in range(numLayers):
            self.net.add_module("Layer %d" % (i+1), Conv(ch, ch, k_size, 1, dilation, bias, separable))
        self.down = Conv(ch, ch*2, k_size, 2, 1, bias)

        self.residual = residual

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y += x
        return self.down(y), y

class TrConvBlock(nn.Module):
    def __init__(self, ch, k_size, numLayers, dilation, bias, residual=False, separable=False):
        super().__init__()

        self.net = nn.Sequential()
        for i in range(numLayers):
            self.net.add_module("Layer %d" % (i+1), Conv(ch*2, ch*2, k_size, 1, dilation, bias, separable))
        self.up = TrConv(ch*2, ch, k_size, 2, 1, bias)

        self.residual = residual

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y += x
        return self.up(x)

class PredictorNet(nn.Module):
    def __init__(self, ch, numOut = 2, levels=5, layers=2, k_size=3, dilation=1, bias=True, residual=False):
        super().__init__()

        self.features = Encoder(ch, levels, layers, k_size, dilation, bias, residual)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(ch*(2**levels), numOut)
        # self.double()

    def forward(self, x):

        features, _ = self.features(x)
        reducedFeatures = self.pool(features).squeeze()
        output = torch.tanh(self.output(reducedFeatures))

        return output

class Encoder(nn.Module):
    def __init__(self, ch, levels=5, layers=2, k_size=3, dilation=1, bias=True, residual=False, separable=False):
        super().__init__()

        self.features = nn.ModuleList()
        self.initial = Conv(4, ch, k_size, 1, 1, bias)
        for i in range(levels):
            self.features.append(ConvBlock(ch*(2**i), k_size, layers, dilation, bias, residual, separable))


    def forward(self, x):

        x = self.initial(x)

        features = []

        for block in self.features:
            x, y = block(x)
            features.append(y)

        return x,features

class Decoder(nn.Module):
    def __init__(self, ch, levels=5, layers=2, k_size=3, dilation=1, bias=True, residual=False, separable=False):
        super().__init__()

        self.features = nn.ModuleList()
        for i in range(levels)[::-1]:
            self.features.append(TrConvBlock(ch*(2**i), k_size, layers, dilation, bias, residual, separable))


    def forward(self, x):

        x, features = x

        for block, y in zip(self.features, features[::-1]):
            x = block(x) + y

        return x

class Segmenter(nn.Module):
    def __init__(self, ch, numOut = 1, levels=5, layers=2, k_size=3, dilation=1, bias=True, residual=True, separable=False):
        super().__init__()

        self.encoder = Encoder(ch, levels, layers, k_size, dilation, bias, residual, separable)
        self.decoder = Decoder(ch, levels, layers, k_size, dilation, bias, residual, separable)
        self.classifier = nn.Conv2d(ch, numOut, 1)

    def forward(self, x):
        features = self.decoder(self.encoder(x))

        predictions = torch.sigmoid(self.classifier(features))

        return features, predictions


'''class Robot(nn.Module):
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
        return loss, image'''