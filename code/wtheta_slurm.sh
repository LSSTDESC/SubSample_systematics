#!/bin/bash -l

#SBATCH --qos regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J no5
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --mail-user=hkong@ifae.es  
#SBATCH --mail-type=ALL
#
module load conda
conda activate base
srun -N 1 -n 1 -c 128 python wtheta_window.py
