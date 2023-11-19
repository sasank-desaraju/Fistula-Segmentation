#!/bin/bash

# This is to create the environment using the mamba solver.

conda create -n fistula python=3.10
conda activate fistula
conda install -y -c conda-forge mamba

# Now install the actual packages
mamba install -y pandas scikit-learn scikit-image matplotlib seaborn
mamba install -y -c conda-forge monai
mamba install -y nibabel
mamba install -y -c SimpleITK SimpleITK
mamba install -y -c conda-forge opencv
mamba install -y -c conda-forge wandb
mamba install -y -c conda-forge albumentations
mamba install -y -c nvidia cuda-python
mamba install -y pytorch torchvision cudatoolkit -c pytorch -c nvidia
mamba install -y -c conda-forge pytorch-lightning

# bruhhh "Torch is not compiled with CUDA enabled"
