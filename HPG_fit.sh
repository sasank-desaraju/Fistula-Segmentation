#!/bin/bash
#SBATCH --account=ezimmer2
#SBATCH --job-name=Fistula-Segmentation
#SBATCH --mail-user=sasank.desaraju@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output ./slurm/logs/my_job-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"


# module load cuda/11.1.0
module load gcc
echo "gcc version is $(gcc --version)"
module load conda
#module load pytorch/1.10
#module load cuda/11.4.3
export PATH=/blue/banks/sasank.desaraju/conda_envs/envs/jtml/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/blue/banks/sasank.desaraju/conda_envs/envs/jtml/lib/
#nvcc --version
# does this let us find the images?

python scripts/fit.py --dataroot /blue/ezimmer2/Fistula_AI/Images/Segmentations/
