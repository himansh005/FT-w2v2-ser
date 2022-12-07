#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it

xent_alpha=$2
ckptdir=$1/alpha_${xent_alpha}
outputdir=outputs/reload_alpha_${xent_alpha}
ckpt=$3
conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

python -u run_downstream_inference.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path ${ckptdir} \
                                              --output_path ${outputdir} \
                                              --nworkers 4 \
                                              --batch_size 8 \
                                              --pretrained_path ${ckpt} \
					--xent_alpha ${xent_alpha}
                                            #   --resume_from_ckpt ${ckpt} \
