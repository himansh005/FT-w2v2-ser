import Levenshtein as lv 
from pathlib import Path  
import json   
import sys
from tqdm import tqdm
if __name__=="__main__":
    if len(sys.argv)<2:
        print('usage: python uer.py cluster_file')
    infile=Path(sys.argv[1])
    base=infile.parent
    ncluster=infile.stem
    # base="clustered_outputs/alpha_0.5"                                                                                                                        
                                                                                                                                    
                                                                                                                                   
                                                                                                                                                
                                                                                                                                
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
                    d[split][label]=[]                                                                                                                        
            d[split][label].append((num,k,spkr,val)) 
    # evalset=[p for p in clus100.keys() if 'evaluation' in p]
    # len(evalset)
    split='train'
    for label in ['Happy','Angry','Neutral','Sad','Surprise']:
            d[split][label].sort(key=lambda x:x[2])
            d[split][label].sort(key=lambda x:x[0])

    #can check inter speaker dist later
    # # spkr1=0,spkr2=9
    # # s1=d[split][label1][spkr1][4]
    # # s2=d[split][label1][spkr2][4]
    labels = ['Happy','Angry','Neutral','Sad','Surprise']
    res = {}
    for i in tqdm(range(len(labels))):
        for j in tqdm(range(i+1,len(labels))):  
            label1=labels[i]
            label2=labels[j]
            res[str((label1,label2))]=0
            assert len(d[split][label1])==len(d[split][label2])
            for uidx in range(len(d[split][label1])):
                s1=d[split][label1][uidx][3]
                s2=d[split][label2][uidx][3]
                if uidx==0:
                        print(label1,label2)
                        print(s1)
                        print(s2)
                dist=lv.distance(s1,s2)
                res[str((label1,label2))]+=dist
            res[str((label1,label2))]/=len(d[split][label2])
    
    with (base/f"dist_{ncluster}.json").open('w') as f:
        json.dump(res,f)
    
    print(res)
    # import readline; print('\n'.join([str(readline.get_history_item(i + 1)) for i in range(readline.get_current_history_length())]))  