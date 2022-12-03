import os
import numpy as np
import json
import random
from pathlib import Path
from glob import glob
import sys

#un-used
labeldict = {
    'ang': 'anger',
    'hap': 'happy',
    'exc': 'happy',
    'sad': 'sad',
    'neu': 'neutral'
}

typedict = {
    'evaluation':'Val',
    'train':'Train',
    'test':'Test'
}

labels = {
    'Train': {},
    'Val': {},
    'Test': {}
}

IEMOCAP_DIR = Path(sys.argv[1])

paths = glob(os.path.join(IEMOCAP_DIR, "*/*/*/*.wav"))

for path in paths:

    path_arr = path.split("/")
    num = int(path_arr[-4])
    if(num<11):
        continue
    type = typedict[path_arr[-2]]
    emotion = path_arr[-3].lower()
    labels[type][path] = emotion
    # break
    # if(len(labels["Train"])>10 and len(labels["Val"])>10 and len(labels["Test"])>10):
    #     break
    
with open(os.path.join(IEMOCAP_DIR, 'labels.json'), 'w') as f:
    json.dump(labels, f, indent=4)
