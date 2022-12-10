#!/bin/bash
# use the bash shell
# set -x 
eval "$(conda shell.bash hook)"
# echo each command to standard out before running it


#best valid UAR checkpoint, trained w pmgn 1, nmgn1 for 11 epochs
step=50000
genroot=generations_multispkr
find -L ${genroot} -type d -name "*${step}" | sed "s/${genroot}\///" | while read -r genconf;do
echo ${genconf}
./run_inference_gen.sh ${genconf}
done