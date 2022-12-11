# 11785 Project 29 Team 32: Improving Emotion Consistency of Speech Resynthesis Through SER finetuning

Codebase for 11-785 Course Project adapted from the official implementation for the paper [Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition](http://arxiv.org/abs/2110.06309). Original README is saved in `original_README.md`

This is only half the repo necessary to run the entire pipeline. For the vocoder stage, please refer to the Vocoder Repo below.

## Running the pipeline
### Installation
The absolute essential packages are placed in `essential_reqs.txt`, we also provide a conda env setup `custom_env_psc` but it's only tested on psc.
### Preprocessing
Download and Place/Softlink the ESD datafolder under this directory, and rename it to `ESD`.

Then, run 
```
python Dataset/IEMOCAP/generate_session_label_wise.py ESD
``` 
to preprocecss the data. This will generate a `labels.json` file which we use for training and inference.
### Run Encoder Training
We provide sample training scripts that you can submit with slurm under `run_contrastive_train_scripts`. The sample command is 
```
python run_downstream_contrastive.py \
                --precision 16 \
                --datadir ./ \
                --labelpath ESD/labels.json \
                --saving_path ${ckptdir} \
                --output_path ${outputdir}\
                --nworkers 4 \
                --batch_size 8 \
				--xent_alpha 0.5
```
You can adjust the xent_alpha parameter for the mixture weight between Cross Entropy loss and Contrastive loss.
This defaults to train for 15 epochs and saving the best checkpoint based on Validation Unweighted Average Recall (UAR).

Changes in our version:
- We added the HuBERT wrapper in:
- We added metric learning using the [pytorch-metric-learning package](https://kevinmusgrave.github.io/pytorch-metric-learning/) in:

### Run K-Means Clustering
We first need to cleanup the checkpoint dictionary keys before we can properly load the checkpoint. This might be due to some untested refactoring or version changes in pytorch lightning in the original repo.
Here is a sample command
```
python ckpt_cleanup.py new_ckpts/alpha_0.5/epoch\=04-valid_loss\=0.000-valid_UAR\=0.88900.ckpt cleaned_ckpts/
```
Then, we perform K-Means clustering. We again provide slurm run scripts in `run_all_cluster.sh` and `run_cluster.sh` The following sample command trains K-means clusters of different sizes and encode the speech into hidden unit sequences (i.e. sequences of the K quantized vector indices) Furthermore, it stores the encoded vectors (not quantized) for each speech sequence as well as the average pooled encoded vectors (always the last layer) used for classification.
```
python cluster.py --model_path ${ckpt_path} \
                    --datadir ./ \
                    --labelpath ESD/labels.json \
                    --outputdir ${outdir}/alpha_${xent_alpha}_L${layer} \
                    --model_type hubert \
                    --sample_ratio 1.0 \
                    --num_clusters "100,1024,2048" \
                    --layer ${layer}
```
Our HuBERT wrapper in :[] allows for passing in a layer parameter to extract embeddings at different levels for clustering. We recommend setting layer to 2 or 6 for best performance.

### Unit Distance and Diversity Heuristics
The following command calculates 1) average unit Levenstein distance between the same speech read in two different emotions, and 2) average unit diversity, which is `#unique units/sequence length`.
Calculating 1) helps us determine whether the same speech has good separation between two emotions.
We also find that 2) unit diversity correlates well with intelligibility of generated speech, so we use it as a heuristic for hyperparameter search.
```
for i in `find ${outdir} -name "*-clus.json" | grep -v all`;do python unit_distance.py $i;done
```

### Create Manifests for vocoder
The following command generates the necessary manifest files for vocoder training and inference.
```
for i in `find ${outdir} -name {num}-clus.json`
do python create_manifest.py --token_store_file $i --outdir ${manifest_dir} \
--downsample 0.35 \
--ds_by_spkr
;done
```
The `--downsample` argument selects of subset of speech for each speaker, each emotion, and each train/dev/test split. When the `--ds_by_spkr` flag is used, it downsamples the number of speakers by the `--downsample` ratio and take the ceiling. We use this as our default setting and used a 4 speaker subset for vocoder training due to computational constraints.



# Vocoder Repo

We adapted the official implementation of the [Speech Resynthesis from Discrete Disentangled Self-Supervised Representations.](https://arxiv.org/abs/2104.00355) paper for our task.
When you've reached this step, you would've created the manifests (which contains the wav file and corresponding hidden units) for all your configurations. You would need to create config files before running training and inference. For more detail, please refer to the
[Vocoder Code here](https://github.com/jeffersonHsieh/speech-resynthesis.git)

We made changes in the model and dataset code for it to take in an additional averaged hidden state embedding, which is then projected to the same dimension as the vocoder's other inputs.

# Evaluation
## Emotional Consistency
Assuming You've trained and resynthesized the audios, you should first softlink/place the generated audio folders under this directory.
Then you need to create the `labels.json` for the resynthesized dataset, which is basically a mapping of filepaths to emotion labels. The following is a sample command that generates the `labels.json`, please refer to the script for what it expects of the input directory.
```
python generate_resynth_labels.py generations_multispkr/ESD_spkr/hubert_nclus_100_alpha0.5_down_sample_035_layer_12/g_00050000 ESD/labels.json
```

We selected the best checkpoint based on UAR in our finetuning to perform SER on the original and resynthesized speech. The script `run_all_inference_gen.sh` and `run_inference_gen.sh` are useful for slurm jobs.
The following is a sample command to generate the predictions. You can use the same labels.json for your ground truth speech, since we provide a filter script to filter out unused data at later stage.
```
python run_downstream_inference.py --precision 16 \
                                        --datadir ./ \
                                        --labelpath ${labels} \
                                        --saving_path ${ckptdir} \
                                        --output_path ${resynth_outputdir} \
                                        --nworkers 4 \
                                        --batch_size 8 \
                                        --pretrained_path ${ckpt} \
                                        --xent_alpha 0.5

```
`${ckpt}` is the checkpoint we use for prediction. The `${ckptdir}` simply saves the confusion matrix image. You will need the ${resynth_outputdir} for later stage, which saves a `predictions.csv` of the output labels.


## Other Synthesis Quality Evaluation
We use ESPNet2's tts official evaluation script for [MCD](link) and [logf0_RMSE](link). Due to a numeric range mismatch issue in the synthesized speech, you will have to change the file read dtype in the evaluation script from int16 to float64 to get the correct results. Please refer to the vocoder repo for the commands.


## Generating Reports

This script gathers results for all configurations

```
python gather_results.py --gen_predictions_dir ${resynth_outputdir} \
                        --gt_predictions ${gt_output_dir}/predictions.csv \
                        --wavscp_dir ${wavscps} \
                        --runid_map map_path_runs.csv
```
You need to pass in the base directory for all your resynthesized SER prediction output, a ground truth (can be unfiltered) predictions.csv (these are all generated at the SER inference stage above), the ${wavscps} is the base directory for all the MCD/logf0 results, and finally, you need a runid to path mapping that maps the slurm-run ids (or your experiment run ids) to the relative subpath of each configuration outputs.

