#!/bin/bash
#SBATCH --job-name=train-virtues
#SBATCH --output=/users/amarti51/logs/train-virtues-PCa-%A.log
#SBATCH --error=/users/amarti51/logs/train-virtues-PCa-%A.err
#SBATCH --account=mrapsoma_prometex

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --mem 16G
#SBATCH --time 12:00:00

# Load modules
module purge
module load python/3.12.1
source /work/FAC/FBM/DBC/mrapsoma/prometex/.venvs/virtues/bin/activate
export PYTHONPATH=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues

python -m src.train_virtues --config_path=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues/external/virtues/configs/config_PCa.yaml