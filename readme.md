Pytorch version 1.9.1 required, do

`pip install torch==1.9.1`

torchvision version 0.10.1 required, do

`pip install torchvision==0.10.1`

pytorch3d version 0.6.1 required, do

`pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt1100/download.html`

or run

`docker pull gaetanlandreau/pytorch3d` <
`sudo docker run --gpus all -it --rm -v ~/occlusionenv:/occlusionenv -w /occlusionenv gaetanlandreau/pytorch3d`
`bash setup_docker.sh`
