#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it


conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

python -u run_downstream_contrastive.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path checkpoints-sb/xent1 \
                                              --output_path outputs \
                                              --nworkers 4 \
                                              --batch_size 8 \
					--xent_alpha 0.5
