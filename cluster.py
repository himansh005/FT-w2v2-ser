from pretrain.trainer import MinimalClassifier
from pretrain.dataloader import MixedDataset
import faiss
from torch.utils import data
import torch
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_clusters', type=str, default='8,64,512,4096')

parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--wav2vecpath', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2','hubert'], default='hubert')
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--sample_ratio', type=float, default=0.2)
parser.add_argument('--unsupdatadir', type=str, default=None)
parser.add_argument('--labelpath', type=str, default=None)
parser.add_argument('--layer', type=int, default=-1)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--outputdir', type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

def train_kmeans(x, nmb_clusters, verbose=False):
    #Subsample data
    n_data, d = x.shape

    
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    clus.niter = 20
    clus.max_points_per_centroid = 500000


    # km = faiss.Kmeans(d,nmb_clusters,niter=20,verbose=True,gpu=True)
    # km.train(x)
    # kmeans.centroids #np array
    # _,I = km.index.search(x) #need reshape (2035817, 1), og vectors are (2035817, 768)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x.astype(np.float32), index)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))
    return index


def eval_kmeans(seq, index):
    seq = np.ascontiguousarray(seq.astype(np.float32))
    _, I = index.search(seq, 1)
    labels = [int(i[0]) for i in I]
    return labels


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model_path:
    model = MinimalClassifier.load_from_checkpoint(args.model_path, strict=False,
                                                   backend=args.model_type, wav2vecpath=args.wav2vecpath)
else:
    model = MinimalClassifier(backend=args.model_type, wav2vecpath=args.wav2vecpath)
model.to(device)
model.freeze()
model.eval()

nclusters = [int(x) for x in args.num_clusters.split(',')]

# First Round Sampling data for k-means cluster estimation
print ('Sampling data for estimation of k-means parameters')
hidden_size = 768 if args.model_type in ['wav2vec2','hubert'] else 512
sampled_reps = np.zeros((max(nclusters) * 1000, hidden_size), dtype=np.float32) #pre-allocate contiguous RAM
head = 0

_data = MixedDataset(args.datadir, args.unsupdatadir, args.labelpath)
kmeans_dataloader = data.DataLoader(_data,
                             batch_size=1,
                             num_workers=8,
                             shuffle=True)
for batch in tqdm(kmeans_dataloader):
    name = batch[1][0]
    batch = batch[0].to(device)
    if args.precision == 16:
        with torch.cuda.amp.autocast():
            representation,avg_rep = model(batch,layer=args.layer)#[0]
    else:
        representation,avg_rep = model(batch,layer=args.layer)#[0]
    representation = representation.squeeze(0).cpu().numpy() #L, C

    #Subsample
    length = representation.shape[0]
    size = max(int(length * args.sample_ratio), 1)
    idx = np.random.choice(length, size=size, replace=False)
    if head+size > len(sampled_reps):
        break
    sampled_reps[head: head+size] = representation[idx]
    head += size


sampled_reps = sampled_reps[:head]
print (f'{head} examples used in k-means...')
print ("Training k-means clustering...")
kmeans_index = []
for nclus in nclusters:
    kmeans_index.append(train_kmeans(sampled_reps, nclus, verbose=True))



#Run inference to get the cluster assignments of all data
print ("Start second phase inference of pseudo-labels")
_data = MixedDataset(args.datadir, args.unsupdatadir, args.labelpath,all=True)
dataloader = data.DataLoader(_data,
                             batch_size=1,
                             num_workers=8,
                             drop_last=False,
                             shuffle=False)
combined_dict = dict()
names = []
reps = []
avg_reps = []
outputdata = [dict() for _ in nclusters]
for batch in tqdm(dataloader):
    name = batch[1][0]
    names.append(name)

    batch = batch[0].to(device)
    if args.precision == 16:
        with torch.cuda.amp.autocast():
            representation,avg_rep = model(batch,layer=args.layer)
    else:
        representation,avg_rep = model(batch,layer=args.layer)
    representation = representation.squeeze(0).cpu().numpy() #L, C
    avg_rep = avg_rep.squeeze(0).cpu().numpy()
    reps.append(representation)
    avg_reps.append(avg_rep)
    combined_dict[name] = []
    for i, index in enumerate(kmeans_index):
        indicies = eval_kmeans(representation, index)
        combined_dict[name].append(indicies)
        outputdata[i][name] = indicies

for i, nclus in enumerate(nclusters):
    with open(os.path.join(args.outputdir, f'{nclus}-clus.json'), 'w') as f:
        json.dump(outputdata[i], f, indent=4)

with open(os.path.join(args.outputdir, 'all-clus.json'), 'w') as f:
    json.dump(combined_dict, f, indent=4)
#dump names & reps zip
rep_dics={}
avg_repdics = {}
for name,rep,avg_rep in zip(names,reps,avg_reps):
    rep_dics[name]=rep #.append({'name':name,'rep':rep})
    avg_repdics[name]=avg_rep #.append({'name':name,'avgrep':avg_rep})
with open(os.path.join(args.outputdir, 'avg_reps_dic.pkl'), 'wb') as f:
    pickle.dump(avg_repdics,f)
with open(os.path.join(args.outputdir, f'all_reps_dic.pkl'), 'wb') as f:
    pickle.dump(rep_dics,f)