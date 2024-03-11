# Ambulance Classifier

## Introduction
Ambulance classifier is CNN-based model that classifies images of ambulances. The model is trained on a dataset of 100 images of ambulance. The model is trained using PyTorch and PyTorch Lightning. The model is trained using a MobileNetV3 architecture.

## Getting Started
### Prerequisites
We assume you have **docker** installed on you machine. If not, you can install it from [here](https://docs.docker.com/get-docker/). Also, you need to have installed the **NVIDIA Container Toolkit** |to run the docker container with GPU support. You can install it from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Training
To train the model we will use [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) to create a development environment. So, you need to have **Visual Studio Code** installed on your machine. If not, you can install it from [here](https://code.visualstudio.com/). Also, you need to have the **Remote - Containers** extension installed on your Visual Studio Code. If not, you can install it from [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Next, you need to clone the repository to your local machine:
```bash
git clone https://github.com/ruhyadi/truck-ambulance-cls
```
Open the repository in VSCode. Press `F1` and type `Remote-Containers: Reopen in Container`. This will open the repository in a devcontainer.

Inside the devcontainer, you able to train the model. The easiest way to train the model is by crate a new experiment in `.yaml` file. For example, can use the `configs/experiment/mobilnetv3.yaml` as reference:
```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /data: ambulance_data
  - override /model: ambulance_model
  - override /callbacks: default
  - override /trainer: gpu

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["mobilenetv3"]

seed: 42069

trainer:
  min_epochs: 5
  max_epochs: 25
```
To train the model, you can run the following command:
```bash
python src/train.py experiment=example
```
The checkpoints and logs will be saved in the `logs` directory.

### Convert to ONNX
To convert the trained model to ONNX format, you can run the following command:
```bash
python tools/torch_to_onnx.py \
    --ckpt_path /path/to/ckpt.pth \
    --backbone mobilenetv3 \
    --categories ambulance \
    --output_path /path/to/output.onnx
```
Your ONNX model will be saved in the `output_path` directory.

## Awknowledgement
- [Hydra](https://hydra.cc/): Hydra is a framework for elegantly configuring complex applications. Developed by Facebook AI Research.
- [PyTorch](https://pytorch.org/): PyTorch is an open source machine learning framework that accelerates the path from research prototyping to production deployment.
- [Lightning](https://lightning.ai/): PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.