from pathlib import Path  
import json   
from tqdm import tqdm
import librosa
import argparse
from multiprocessing import Pool


def get_duration(filepath):
    return librosa.get_duration(filename=filepath)


def save(outdir, tr, cv, tt):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f'train.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tr]))
    with open(outdir / f'val.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in cv]))
    with open(outdir / f'test.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tt]))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',type=str,choices = ['ESD','ESD_correct'],default='ESD_correct')
    parser.add_argument('--token_store_file', type=Path, required=True)
    parser.add_argument('--nworkers',type=int, default=4)
    parser.add_argument('--outdir',type=Path, required=True)
    parser.add_argument('--min_dur', type=float, default=None)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--downsample',type=float,default=0.35)
    parser.add_argument('--ds_by_spkr',type=bool,default=True)
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--tt', type=float, default=0.05)
    # parser.add_argument('--cv', type=float, default=0.05)
    args = parser.parse_args()


    infile=args.token_store_file
    base=infile.parent
    ncluster=infile.stem 
    args.outdir.mkdir(parents=True,exist_ok=True)
    alpha = 'alpha_'+args.token_store_file.parent.name.split('_')[1].replace('_','').replace('.','')
    layer = args.token_store_file.parent.name.split('_')[2].replace('L','')
    configs = f"nclus_{args.token_store_file.stem.split('-')[0]}_{alpha}_down_sample_{str(args.downsample).replace('.','')}_layer_{layer}"
    with (infile).open() as f:                                                                                                                  
            clus=json.load(f)                                                                                                                              
                                                                                                                                          
    d = {}     
    for k in clus:                                                                                                                                         
            karr=k.split('/')                                                                                                                                 
            _,spkr,label,split,name=karr
            uttid=Path(name).stem                                                                                                                      
            val=clus[k]                                                                                                                                    
            if split not in d:                                                                                                                                
                d[split]={}                                                                                                                               
            num=int(name.split('.')[0].split('_')[1])                                                                                                         
            if label not in d[split]:                                                                                                                         
                d[split][label]={}
            if spkr not in d[split][label]:
                d[split][label][spkr] = []                                                                                                                    
            d[split][label][spkr].append((num,k,spkr,val)) 
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
    get_hrs = lambda l:sum([x['duration'] for x in l])
    for split in ['train','evaluation','test']:
        for label in ['Happy','Angry','Neutral','Sad','Surprise']:
                d[split][label].sort(key=lambda x:x[2])
                d[split][label].sort(key=lambda x:x[0])

        #can check inter speaker dist later
        # # spkr1=0,spkr2=9
        # # s1=d[split][label1][spkr1][4]
        # # s2=d[split][label1][spkr2][4]
        labels = ['Happy','Angry','Neutral','Sad','Surprise']
        units = []
        paths = []
        for i in range(len(labels)):
            label1=labels[i]
            for uidx in range(len(d[split][label1])):
                units.append(d[split][label1][uidx][3])
                paths.append(d[split][label1][uidx][1])
    
        with Pool(args.nworkers) as p:
            durations = list(tqdm(p.imap(get_duration, paths), total=len(paths)))
        
        for code,path,dur in zip(units,paths,durations):
            sample = {}
            sample['audio'] = str(path)
            sample['hubert'] = ' '.join(list(map(str,code)))
            sample['duration'] = int(dur) #/ 16000
            outdict[split].append(sample)
                
        print(f"{split} num samples: {len(outdict[split])}")
        print(f"{split} num hrs: {get_hrs(outdict[split])/3600}")

    # for fname,dur in tqdm(fname_durs):

    #     sample = {}
    #     uttid = os.path.splitext(os.path.basename(fname))[0]
    #     code = uttid2code[uttid]



    #     if args.min_dur and sample['duration'] < args.min_dur:
    #         continue

    #     samples += [sample]
    od=outdict
    if args.ds_by_spkr:
        tag=args.tag+'_spkr'
    else:
        tag=args.tag
    # save(args.outdir/tag/f'hubert_{configs}', od["train"], od["evaluation"], od["test"])