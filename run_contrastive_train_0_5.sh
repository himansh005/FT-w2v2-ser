#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it

xent_alpha=0.5
ckptdir=/ocean/projects/tra220029p/chsieh1/emo_checkpoints/alpha_${xent_alpha}_retr
outputdir=/ocean/projects/tra220029p/chsieh1/emo_outputs/alpha_${xent_alpha}_retr
# if [ -e ${ckptdir} ];then echo "exists";exit 1;fi
#  if [ -e ${outputdir} ];then echo "exists";exit 1;fi
conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

python run_downstream_contrastive.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path ${ckptdir} \
                                              --output_path ${outputdir}\
                                              --nworkers 4 \
                                              --batch_size 8 \
					--xent_alpha ${xent_alpha}
