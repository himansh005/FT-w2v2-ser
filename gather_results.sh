#!/bin/bash
python gather_results.py --gen_predictions_dir emo_eval_outputs \
                        --gt_predictions emo_eval_outputs/gt_all/predictions.csv \
                        --wavscp_dir wavscps \
                        --runid_map ../speech-resynthesis/map_path_runs.csv \
