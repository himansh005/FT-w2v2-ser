import Levenshtein as lv 
from pathlib import Path  
import json   
import sys
from tqdm import tqdm
import librosa
import argparse
from multiprocessing import Pool
import pickle




def save(outdir, tr, cv, tt):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f'train.pkl', 'wb') as f:
        pickle.dump(tr,f)
    # with open(outdir / f'val.pkl', 'wb') as f:
    #     pickle.dump(cv,f)
    # with open(outdir / f'test.pkl', 'wb') as f:
    #     pickle.dump(tt,f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_store_dir', type=Path, required=True)
    parser.add_argument('--outdir',type=Path, required=True)
    parser.add_argument('--downsample',type=float,default=0.1)
    parser.add_argument('--ds_by_spkr',type=bool,default=True)
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--tt', type=float, default=0.05)
    # parser.add_argument('--cv', type=float, default=0.05)
    args = parser.parse_args()


    base=args.token_store_dir
    args.outdir.mkdir(parents=True,exist_ok=True)
    avg_token_file=base/'avg_reps_dic.pkl'
    all_token_store=base/'all_reps_dic.pkl'
    alpha = base.name.split('_')[1]
    layer = base.name.split('_')[2]
    configs = f"alpha_{alpha}_down_sample_{str(args.downsample).replace('.','')}_layer_{layer}"
                                                                                                                                
    with (all_token_store).open('rb') as f:                                                                                                                  
        all_tok_dics=pickle.load(f)                                                                                                                             
    with (avg_token_file).open('rb') as f:                                                                                                                  
        avg_tok_dics=pickle.load(f)                                                                                                                                           
    d = {}     
    for k in all_tok_dics:                                                                                                                                         
            rep=all_tok_dics[k]
            avg_rep=avg_tok_dics[k]
            karr=k.split('/')
            _,spkr,label,split,name=karr
            uttid=Path(name).stem                                                                                                                      
            if split not in d:                                                                                                                                
                d[split]={}                                                                                                                               
            num=int(name.split('.')[0].split('_')[1])                                                                                                         
            if label not in d[split]:                                                                                                                         
                d[split][label]={}
            if spkr not in d[split][label]:
                d[split][label][spkr] = []                                                                                                                    
            d[split][label][spkr].append((num,k,spkr,rep,avg_rep)) 
    num_spkr = len(d[split][label])
    if args.downsample < 1:
        if args.ds_by_spkr:
            num_spkr=min(int(num_spkr*args.downsample)+1,num_spkr)
        for split in d:
            for label in d[split]:
                if not args.ds_by_spkr:
                    for i,spkr in enumerate(d[split][label]):
                        ol = d[split][label][spkr]
                        dsl = int(args.downsample*len(ol))
                        d[split][label][spkr] = ol[:dsl]
                    od = d[split][label]
                    d[split][label] = [x for spkrl in od.values() for x in spkrl]
                else:
                    od = d[split][label]
                    d[split][label] =  [x for i,spkrl in enumerate(od.values()) for x in spkrl if i<num_spkr]
    

    # evalset=[p for p in clus100.keys() if 'evaluation' in p]
    # len(evalset)
    outdict = {
        "train":[],
        "evaluation":[],
        "test":[]
    }

    for split in ['train','evaluation','test']:
        for label in ['Happy','Angry','Neutral','Sad','Surprise']:
                d[split][label].sort(key=lambda x:x[2])
                d[split][label].sort(key=lambda x:x[0])

        #can check inter speaker dist later
        # # spkr1=0,spkr2=9
        # # s1=d[split][label1][spkr1][4]
        # # s2=d[split][label1][spkr2][4]
        labels = ['Happy','Angry','Neutral','Sad','Surprise']

        for i in range(len(labels)):
            label1=labels[i]
            for uidx in range(len(d[split][label1])):
                num,k,spkr,reps,avg_rep = d[split][label1][uidx]
                outdict[split].append((k,reps,avg_rep))
    
    

        print(f"{split} num samples: {len(outdict[split])}")

    # for fname,dur in tqdm(fname_durs):

    #     sample = {}
    #     uttid = os.path.splitext(os.path.basename(fname))[0]
    #     code = uttid2code[uttid]



    #     if args.min_dur and sample['duration'] < args.min_dur:
    #         continue

    #     samples += [sample]
    od=outdict
    save(args.outdir/f'hubert_{configs}_reps_ds_spkr', od["train"], od["evaluation"], od["test"])
    print("done")