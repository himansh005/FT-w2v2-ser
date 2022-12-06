. paths.sh

python -u run_downstream_custom.py --precision 16 \
                                              --datadir ./ \
                                              --labelpath ESD/labels.json \
                                              --saving_path checkpoints \
                                              --output_path outputs \
                                              --nworkers 4 \
                                              --batch_size 16
