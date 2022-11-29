#!/bin/bash

#export PATH=~/anaconda3/bin:$PATH
#conda init bash
#conda activate jtml

export PATH=/blue/banks/sasank.desaraju/conda_envs/jtml/bin/:$PATH
export LD_LIBRARY_PATH=/blue/banks/sasank.desaraju/conda_envs/jtml/lib/:$LD_LIBRARY_PATH

python scripts/fit.py --dataroot '/blue/ezimmer2/Fistula_AI/Images/Segmentations/'
