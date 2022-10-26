# Setting up the environment

## Local environment

For local environment from `requirements.txt` using conda do:

`conda create --name OcclusionEnv --file requirements.txt`

## Using docker

`docker pull gaetanlandreau/pytorch3d`

`sudo docker run --gpus all -it --rm -v local/path/to/repository:/occlusionenv -w /occlusionenv gaetanlandreau/pytorch3d`

Note: change `local/path/to/repository` to the location where you cloned this repo.

`bash setup_docker.sh`
