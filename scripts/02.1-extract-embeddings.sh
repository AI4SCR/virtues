#!/usr/bin/env bash

cd /Users/adrianomartinelli/projects/esm || exit

source ~/.venv/esm/bin/activate
EMBEDDINGS_DIR=$HOME/data/virtues/embeddings/esm2_t30_150M_UR50D
python scripts/extract.py esm2_t30_150M_UR50D ~/data/virtues/datasets/PCa/protein-sequences.fasta \
 $EMBEDDINGS_DIR --repr_layers 30 --include mean
deactivate

source ~/.venv/ai4bmr-imc/bin/activate
python /Users/adrianomartinelli/projects/ai4bmr/BLCaPCa/workflow/scripts/virtues/02.0-convert-embed-dict-to-tensor.py \
--embeddings_dir=$EMBEDDINGS_DIR



