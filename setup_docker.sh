#!/usr/bin/env sh
pip uninstall pytorch3d
pip install opencv-python-headless
pip install gym
pip install -U torch
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt1100/download.html