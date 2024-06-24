# EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer


## Installation

This code was tested on Ubuntu 20.04 with CUDA 11.1.

**a. Create a conda virtual environment and activate it.**

```bash
conda create -n elegant python=3.8
conda activate elegant
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

**c. Install other required libaries.**

```bash
pip install opencv-python matplotlib fvcore
```

**d. Install dlib.**

First we need to install cmake

```bash
sudo apt-get update
sudo apt-get install cmake
```

Then we would install dlib

via pip (slow)

```bash
pip install dlib
```

via Conda

```bash
conda install -c conda-forge dlib
```

## Setup

Clone this repository and prepare the dataset and weights through the following steps:

**a. Prepare model weights for face detection.**

Download the weights of [dlib](https://github.com/davisking/dlib) face detector of 68 landmarks [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Unzip it and move it to the directory `./faceutils/dlibutils`.

Download the weights of BiSeNet ([PyTorch implementation](https://github.com/zllrunning/face-parsing.PyTorch)) for face parsing [here](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812). Rename it as `resnet.pth` and move it to the directory `./faceutils/mask`.

**b. Prepare Makeup Transfer (MT) dataset.**

Download raw data of the MT Dataset and unzip it into sub directory `./data`.

Run the following command to preprocess data:

```bash
python training/preprocess.py
```

Your data directory should look like:

```text
data
└── MT-Dataset
    ├── images
    │   ├── makeup
    │   └── non-makeup
    ├── segs
    │   ├── makeup
    │   └── non-makeup
    ├── lms
    │   ├── makeup
    │   └── non-makeup
    ├── makeup.txt
    ├── non-makeup.txt
    └── ...
```

## Train

To train a model from scratch, run

```bash
python scripts/train.py --save_path="./results/elegant/checkpoint-10"
```

## Test

To test our model, run

```bash
python scripts/demo.py
```
