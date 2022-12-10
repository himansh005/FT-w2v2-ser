#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it

xent_alpha=1
ckptdir=checkpoints/alpha_${xent_alpha}_rt
outputdir=outputs/alpha_${xent_alpha}_rt
# if [ -e ${ckptdir} ];then echo "exists";exit 1;fi
#  if [ -e ${outputdir} ];then echo "exists";exit 1;fi
conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

python -u run_downstream_contrastive.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path ${ckptdir} \
                                              --output_path ${outputdir}\
                                              --nworkers 4 \
                                              --batch_size 8 \
					--xent_alpha ${xent_alpha}
