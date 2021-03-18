#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks=8
#SBATCH --mem=16g
#SBATCH --tmp=16g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=girdh005@umn.edu
#SBATCH --gres=gpu:v100:1
#SBATCH -p v100
cd /home/jusun/girdh005/sell-it
export PATH=/home/jusun/girdh005/anaconda3/envs/sellit/bin:$PATH
conda activate sellit
python train.py
