#!/bin/bash
#SBATCH --job-name=virtues-v2
#SBATCH --output=/users/amarti51/logs/virtues-v2-PCa-%A.log
#SBATCH --error=/users/amarti51/logs/virtues-v2-PCa-%A.err
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
source /work/FAC/FBM/DBC/mrapsoma/prometex/.venvs/mae-virtues/bin/activate

export PYTHONPATH=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues

cd /work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues/external/virtues || exit

python -m src.train_virtues --config_path=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/virtues/external/virtues/configs/config-PCa-v2.yaml