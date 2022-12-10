#!/bin/bash
# use the bash shell
# set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it


#best valid UAR checkpoint, trained w pmgn 1, nmgn1 for 11 epochs
ckptdir=emo_eval_ckpts/gt_all
outputdir=emo_eval_outputs/gt_all
ckpt=new_ckpts/alpha_0.5/epoch=11-valid_loss=0.000-valid_UAR=0.90000.ckpt
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
					--xent_alpha 0.5
                                            #   --resume_from_ckpt ${ckpt} \
