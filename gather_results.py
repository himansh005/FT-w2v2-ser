"""
Author: Chia-Chun Hsieh
generates report for all experiments
1. expects gen_predictions_dir,wavscp_dir to follow the same directory structure
where the subdir paths from these two base directories should be in the `runid_map` file
(which stores the runid->subdir mapping)
2. wavscp_dir should contain logf0/mcd results calculated with espnet tts evaluation scripts
3. expects subdirs of gen_predictions_dir for each run configuration 
to contain predictions.csv with columns 'file', 'true', 'predicted'
"""
import pandas as pd
from utils.metrics import ConfusionMetrics
import argparse
from pathlib import Path
import glob

def read_txt(path):
    with open(str(path)) as f:
        res = f.readlines()
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_predictions_dir', type=Path, required=True)
    parser.add_argument('--gt_predictions',type=Path,required=True)
    parser.add_argument('--wavscp_dir',type=Path,required=True) #pre-calculated logf0/mcd results
    parser.add_argument('--runid_map', type=Path, default=None)
    parser.add_argument('--save_dir', type=Path, default=Path('./'))
    parser.add_argument('--format', type=str, choices=['latex','csv'],default='csv')
    # parser.add_argument('--step', type=int,default=50000)
    parser.add_argument('--n_classes', type=int,default=5)
    args = parser.parse_args()
    mapping=pd.read_csv(args.runid_map)
    metrics = ConfusionMetrics(n_classes=args.n_classes)
    dfmap = {}
    f0map = {}
    uarmap = {}
    warmap = {}
    f1macromap = {}
    f1micromap = {}
    mcdmap = {}
    pathmap = {}
    df=pd.read_csv(args.gt_predictions)
    gt_filterd=False
    for i,(key,confdir) in mapping.iterrows():
        pathmap[key]=confdir
        dfmap[key]=pd.read_csv(args.gen_predictions_dir/confdir.strip(' ')/"predictions.csv")
        if not gt_filterd:
            gt_filtered=True
            s = set(dfmap[key]['file'].apply(lambda x:Path(x).name.rsplit('_',1)[0]))
            df=df[df['file'].apply(lambda x:Path(x).name.split('.')[0]).isin(s)]

        metrics.fit(dfmap[key]['true'],dfmap[key]['predicted'])
        uarmap[key] = metrics.uar
        warmap[key] = metrics.war
        f1macromap[key] = metrics.macroF1
        f1micromap[key] = metrics.microF1
        metrics.clear()

        f0res = read_txt(args.wavscp_dir/confdir.strip(' ')/'log_f0_rmse_avg_result.txt')[1].split(' ',1)[1]
        f0map[key] = f0res
        mcdres = read_txt(args.wavscp_dir/confdir.strip(' ')/'mcd_avg_result.txt')[1].split(' ',1)[1]
        mcdmap[key] = mcdres
    metrics.fit(df['true'],df['predicted'])
    uarmap['gt'] = metrics.uar
    warmap['gt'] = metrics.war
    f1macromap['gt'] = metrics.macroF1
    f1micromap['gt'] = metrics.microF1
    results = pd.DataFrame.from_dict({'uar':uarmap,'war':warmap,
        'macroF1':f1macromap,'microF1':f1micromap,'logf0':f0map,'mcd':mcdmap,'path':pathmap})
    
    print(results[results.columns[:-1]])
    results.to_csv(str(args.save_dir/'gatherd_resynth_results.csv'))
    

    
