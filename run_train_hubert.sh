#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it


conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

ckptdir=/ocean/projects/tra220029p/chsieh1/emo_checkpoints/alpha_1
outputdir=/ocean/projects/tra220029p/chsieh1/emo_outputs/alpha_1

python -u run_downstream_custom.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path ${ckptdir} \
                                              --output_path ${outputdir}\
                                              --nworkers 4 \
                                              --batch_size 8 \
                                              --max_epochs 15
