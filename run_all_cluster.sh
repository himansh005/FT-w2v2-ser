#!/bin/bash
# ./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5/epoch\=11-valid_loss\=0.000-valid_UAR\=0.90000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 2
# ./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5/epoch\=11-valid_loss\=0.000-valid_UAR\=0.90000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 6
# ./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5/epoch\=11-valid_loss\=0.000-valid_UAR\=0.90000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 12
# ./run_cluster.sh 1 cleaned_ckpts/alpha_1_notr/epoch=00-valid_loss=1.613-valid_UAR=0.20000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 2
# ./run_cluster.sh 1 cleaned_ckpts/alpha_1_notr/epoch=00-valid_loss=1.613-valid_UAR=0.20000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 6
# ./run_cluster.sh 1 cleaned_ckpts/alpha_1_notr/epoch=00-valid_loss=1.613-valid_UAR=0.20000.ckpt $PROJECT/emo_clustered_w_avgrep/fix 12
./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5_correct/epoch=05-valid_loss=0.000-valid_UAR=0.87600.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 2 
./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5_correct/epoch=05-valid_loss=0.000-valid_UAR=0.87600.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 6
./run_cluster.sh 0.5 cleaned_ckpts/alpha_0.5_correct/epoch=05-valid_loss=0.000-valid_UAR=0.87600.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 12

./run_cluster.sh 0.7 cleaned_ckpts/alpha_0.7_correct/epoch=04-valid_loss=0.000-valid_UAR=0.88900.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 2
./run_cluster.sh 0.7 cleaned_ckpts/alpha_0.7_correct/epoch=04-valid_loss=0.000-valid_UAR=0.88900.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 6
./run_cluster.sh 0.7 cleaned_ckpts/alpha_0.7_correct/epoch=04-valid_loss=0.000-valid_UAR=0.88900.ckpt $PROJECT/emo_clustered_w_avgrep/pm0 12
