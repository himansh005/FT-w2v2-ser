import os
import argparse
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.outputlib import WriteConfusionSeaborn
import torch
from pytorch_lightning.loggers import CSVLogger

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--maxseqlen', type=float, default=10)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
    parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--labelpath', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2','hubert'], default='hubert')

    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--cpu_torchscript', action='store_true')
    parser.add_argument('--xent_alpha',type=float, default=0.5)
    parser.add_argument('--resume_from_ckpt',type=str,default=None)
    # TODO: allowed combo of miner+losses
    # parser.add_argument('--use_miner',type=bool,default=False)
    # parser.add_argument('--metric_loss_type',type=str, choices=['contrastive','triplet'],default=None)
    # parser.add_argument('--miner_strategy',type=str, choices=['hard','semihard','easy'],default=None)

    args = parser.parse_args()
    hparams = args
    print(os.getcwd())
    from downstream.Custom.trainer import DownstreamMetricLearning

    if not os.path.exists(hparams.saving_path):
        os.makedirs(hparams.saving_path)
    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)

    model = DownstreamMetricLearning(hparams)
    if hparams.pretrained_path is not None:
        model = model.load_from_checkpoint(hparams.pretrained_path, strict=False,hparams=hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.saving_path,
        filename='{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if hasattr(model, 'valid_met') else None,
        save_top_k=args.save_top_k if hasattr(model, 'valid_met') else 0,
        verbose=True,
        save_weights_only=True,
        monitor='valid_UAR' if hasattr(model, 'valid_met') else None,
        mode='max'
    )
    # if hparams.resume_from_ckpt is None:
    #     print("did not pass in trained checkpoint!")
    trainer = Trainer(
        amp_backend='native',
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=None,#hparams.resume_from_ckpt,
        check_val_every_n_epoch=1,
        max_epochs=hparams.max_epochs,
        devices=1,
        accelerator='gpu'
    )
    # trainer.fit(model)

    if hasattr(model, 'test_met'):
        trainer.test(model)
        met = model.test_met
        print("+++ SUMMARY +++")
        for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'),
                            (met.uar*100, met.war*100, met.macroF1*100, met.microF1*100)):
            print(f"Mean {nm}: {np.mean(metric):.2f}")
            print(f"Std. {nm}: {np.std(metric):.2f}")
        WriteConfusionSeaborn(
            met.m,
            model.dataset.emoset,
            os.path.join(args.saving_path, 'confmat.png')
        )

if __name__=="__main__":
    main()