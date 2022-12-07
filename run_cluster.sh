#!/bin/bash
# use the bash shell
set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it

conda activate /ocean/projects/cis220078p/chsieh1/miniconda3/envs/ser

. paths.sh

xent_alpha=$1
ckpt_path=$2

python -m pdb cluster.py --model_path ${ckpt_path} \
                    --datadir ./ \
                    --labelpath ESD/labels.json \
                    --outputdir clustered_outputs/alpha_${xent_alpha} \
                    --model_type hubert \
                    --sample_ratio 1.0 \
                    --num_clusters "100,200,500"