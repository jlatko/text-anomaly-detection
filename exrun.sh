#!/usr/bin/env bash
set -e

CONDA_ENV="vae"

# GPU to use
GPU='0'
#
PARAM_UPDATES="with
    'source=friends-corpus'
    'ood_source=parliament-corpus'
    'tags=friends vs parliament scale=0.2 bs=32 \n'
    "
# Execute
FLAG_EXPERIMENT_NAME='--name "experiment"'
CMD="CUDA_VISIBLE_DEVICES=$GPU /home/$USER/anaconda3/envs/$CONDA_ENV/bin/python experiment.py"
EXEC="$CMD $PARAM_UPDATES $FLAG_EXPERIMENT_NAME"
echo executing $EXEC ...
eval $EXEC


PARAM_UPDATES="with
    'source=supreme-corpus'
    'ood_source=friends-corpus'
    'tags=supreme vs friends scale=0.2 bs=32 \n'
    "
# Execute
FLAG_EXPERIMENT_NAME='--name "experiment"'
CMD="CUDA_VISIBLE_DEVICES=$GPU /home/$USER/anaconda3/envs/$CONDA_ENV/bin/python experiment.py"
EXEC="$CMD $PARAM_UPDATES $FLAG_EXPERIMENT_NAME"
echo executing $EXEC ...
eval $EXEC

#env CUDA_VISIBLE_DEVICES=8 /home/$USER/anaconda3/envs/vae/bin/python lang_model_run.py
#env CUDA_VISIBLE_DEVICES=8 /home/$USER/anaconda3/envs/vae/bin/python experiment.py