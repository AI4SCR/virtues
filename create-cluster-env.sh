Sinteractive -m 4G -G 1

module load python/3.12.1
python -m venv ~/.venvs/virtues
source ~/.venvs/virtues/bin/activate

cd ~/projects/virtues || exit
pip install -r requirements.txt