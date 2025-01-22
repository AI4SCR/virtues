#!/bin/bash
#SBATCH --job-name=remote-session
#SBATCH --output=/users/amarti51/logs/remote-session-%A.log
#SBATCH --error=/users/amarti51/logs/remote-session-%A.err
#SBATCH --account=mrapsoma_prometex
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Load modules
module load python/3.11.7
source /work/FAC/FBM/DBC/mrapsoma/prometex/.venvs/imc/bin/activate

PROJECT_DIR=/work/FAC/FBM/DBC/mrapsoma/prometex/projects/BLCaPCa
HOST=curnagl.dcsr.unil.ch

/users/amarti51/.cache/JetBrains/RemoteDev/dist/6be7d8bc400d7_pycharm-professional-243.23654.74/bin/remote-dev-server.sh \
run \
$PROJECT_DIR