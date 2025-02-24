#!/bin/bash
#SBATCH --job-name=virtues-inference
#SBATCH --output=/users/amarti51/logs/virtues-inference-PCa-%A.log
#SBATCH --error=/users/amarti51/logs/virtues-inference-PCa-%A.err
#SBATCH --account=mrapsoma_prometex

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 02:00:00

# Load modules
module purge
module load python/3.12.1
source /work/FAC/FBM/DBC/mrapsoma/prometex/.venvs/virtues/bin/activate
export PYTHONPATH=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues

python scribble/inference.py