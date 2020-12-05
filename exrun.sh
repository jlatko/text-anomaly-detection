#!/usr/bin/env bash
set -e

CONDA_ENV="vae"

# GPU to use
GPU='8'
#
#PARAM_UPDATES="with
#    'source=parliament-corpus'
#    'word_embedding_size=300'
#    "
# Execute
FLAG_EXPERIMENT_NAME='--name "experiment"'
CMD="CUDA_VISIBLE_DEVICES=$GPU /home/$USER/anaconda3/envs/$CONDA_ENV/bin/python experiment.py"
EXEC="$CMD $PARAM_UPDATES $FLAG_EXPERIMENT_NAME"
echo executing $EXEC ...
eval $EXEC

#env CUDA_VISIBLE_DEVICES=8 /home/$USER/anaconda3/envs/vae/bin/python lang_model_run.py
#env CUDA_VISIBLE_DEVICES=8 /home/$USER/anaconda3/envs/vae/bin/python experiment.py