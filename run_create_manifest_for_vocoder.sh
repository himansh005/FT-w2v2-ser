#!/bin/bash

for i in `find emo_clustered_w_avg/pm0/ -name 1024-clus.json`
do python create_manifest.py --token_store_file $i --outdir /ocean/projects/tra220029p/chsieh1/speech-resynthesis/manifests;done
