import pickle
import argparse
from pathlib import Path
import os
from tqdm import tqdm

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_store_dir', type=Path, required=True)

    args = parser.parse_args()

    base=args.token_store_dir
    avg_token_file=base/'avg_reps.pkl'
    all_token_store=base/'all_reps.pkl'
    with (all_token_store).open('rb') as f:                                                                                                                  
        all_tok_dics=pickle.load(f)   
    with (avg_token_file).open('rb') as f:                                                                                                                  
        avg_tok_dics=pickle.load(f)   
    rep_dics={}
    avg_repdics = {}
    for d1,d2 in tqdm(zip(all_tok_dics,avg_tok_dics),total=len(all_tok_dics)):
        name=d1['name']
        rep=d1['rep']
        avg_rep = d2['avgrep']
        rep_dics[name]=rep #.append({'name':name,'rep':rep})
        avg_repdics[name]=avg_rep #.append({'name':name,'avgrep':avg_rep})
    print(len(rep_dics))
    with open(os.path.join(args.token_store_dir, 'avg_reps_dic.pkl'), 'wb') as f:
        pickle.dump(avg_repdics,f)
    with open(os.path.join(args.token_store_dir, f'all_reps_dic.pkl'), 'wb') as f:
        pickle.dump(rep_dics,f)