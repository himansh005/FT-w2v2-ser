import copy
import torch
import sys
from pathlib import Path
if __name__=="__main__":
    if len(sys.argv)<3:
        print('pass ckpt path!')
        exit(1)
    ckptp = Path(sys.argv[1])
    ver = ckptp.parent.name
    name = ckptp.name
    outdir = Path(sys.argv[2])/ver
    outdir.mkdir(exist_ok=True,parents=True)
    ckpt=torch.load(str(ckptp))
        # 'new_ckpts/alpha_0.5/epoch=11-valid_loss=0.000-valid_UAR=0.90000.ckpt')
    state_dict=ckpt['state_dict']

    state_dict_v2 = copy.deepcopy(state_dict)

    for key in state_dict:
            if 'model.' in key:
                    spkeys = key.split('.')
                    nkey='.'.join(spkeys[1:])
                    k = state_dict_v2.pop(key)
                    state_dict_v2[nkey]=state_dict[key]
    torch.save({
        'state_dict':state_dict_v2
        },str(outdir/name))
