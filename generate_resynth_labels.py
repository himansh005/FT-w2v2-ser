import os
import numpy as np
import json
import random
from pathlib import Path
from glob import glob
import sys
import re
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
ref_labels_path = Path(sys.argv[2])
with ref_labels_path.open() as f:
    ref_labels = json.load(f)
paths = glob(os.path.join(IEMOCAP_DIR, "*.wav"))
ptrn = re.compile(r"(?<!_[0-3])_gen.wav$")
paths = [path for path in paths if ptrn.search(path)]
tlabs = {Path(k).name.split('.')[0]:v for k,v in ref_labels['Test'].items()}
for path in paths:
    uttid = Path(path).name.rsplit('_',1)[0]

    type = 'Test'
    emotion = tlabs[uttid]
    labels[type][path] = emotion
    # break
    # if(len(labels["Train"])>10 and len(labels["Val"])>10 and len(labels["Test"])>10):
    #     break
    
with open(os.path.join(IEMOCAP_DIR, 'labels.json'), 'w') as f:
    json.dump(labels, f, indent=4)