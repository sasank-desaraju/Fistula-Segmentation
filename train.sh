#!/bin/bash

#export PATH=~/anaconda3/bin:$PATH
#conda init bash
#conda activate jtml

export PATH=~/miniconda3/envs/jtml/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/envs/jtml/lib/:$LD_LIBRARY_PATH

python scripts/fit.py --dataroot '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/'
