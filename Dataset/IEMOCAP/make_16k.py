import librosa
import os
import soundfile as sf
from pathlib import Path
import sys
from glob import glob

Path('Audio_16k').mkdir(exist_ok=True)
IEMOCAP_DIR = Path(sys.argv[1])
print ("Downsampling ESD to 16k")

paths = glob(os.path.join(IEMOCAP_DIR, "*/*/*/*.wav"))

for full_audio_name in paths:
    audio, sr = librosa.load(str(full_audio_name), sr=None)
    
    if(sr != 16000):
        print("mismatch: ", sr)
        break

#ESD is already at 16kHz, the above is a sanity check only.
