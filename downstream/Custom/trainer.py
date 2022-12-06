import argparse
import os
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_metric_learning import losses as pml_losses

from .dataloader import CustomEmoDataset
from utils.metrics import ConfusionMetrics

from pretrain.trainer import PretrainedRNNHead, PretrainedEmoFeatureHead
from tqdm import tqdm
import csv

class DownstreamGeneral(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hp = hparams
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)
        if self.hp.pretrained_path is not None:
            self.model = PretrainedRNNHead.load_from_checkpoint(self.hp.pretrained_path, strict=False,
                                                                n_classes=self.dataset.nemos,
                                                                backend=self.hp.model_type)
        else:
            self.model = PretrainedRNNHead(n_classes=self.dataset.nemos,
                                           backend=self.hp.model_type)
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor(
            [counter[c] for c in self.dataset.emoset]
        ).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(
            f"Weigh losses by prior distribution of each class: {weights}."
        )

        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        with open(os.path.join(hparams.output_path,'predictions.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["file", "true", "predicted"])
            del writer
        
        # Define metrics
        if hasattr(self.dataset, 'val_dataset'):
            self.valid_met = ConfusionMetrics(self.dataset.nemos)
        if hasattr(self.dataset, 'test_dataset'):
            self.test_met = ConfusionMetrics(self.dataset.nemos)

    def forward(self, x, length):
        return self.model(x, length)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.trainable_params(), lr=self.hp.lr)
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(dataset=self.dataset.train_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=True,
                            num_workers=self.hp.nworkers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if not hasattr(self.dataset, 'val_dataset'):
            return
        loader = DataLoader(dataset=self.dataset.val_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=False,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(dataset=self.dataset.test_dataset,
                            batch_size=1,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def training_step(self, batch, batch_idx):
        feats, length, label, fname  = batch
        pout = self(feats, length)
        loss = self.criterion(pout, label)
        tqdm_dict = {'loss': loss}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, length, label, fname  = batch
        pout = self(feats, length)
        loss = self.criterion(pout, label)
        for l, p in zip(label, pout):
            self.valid_met.fit(int(l), int(p.argmax()))
        self.log('valid_loss', loss, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        print (self.valid_met.uar)
        self.log('valid_UAR', self.valid_met.uar)
        self.log('valid_WAR', self.valid_met.war)
        self.log('valid_macroF1', self.valid_met.macroF1)
        self.log('valid_microF1', self.valid_met.microF1)
        self.valid_met.clear()

    def test_step(self, batch, batch_idx):
        feats, label, fname = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        pout = self(feats, length)
        prediction = int(pout.argmax())
        
        arr = []
        for i in range(len(label)):
            arr.append([fname[i], int(label[i]), prediction])

        with open(os.path.join(self.hp.output_path,'predictions.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(arr)
            del writer

        self.test_met.fit(int(label), prediction)

    def on_test_epoch_end(self):
        """Report metrics."""
        self.log('test_UAR', self.test_met.uar, logger=True)
        self.log('test_WAR', self.test_met.war, logger=True)
        self.log('test_macroF1', self.test_met.macroF1, logger=True)
        self.log('test_microF1', self.test_met.microF1, logger=True)

        print(f"""++++ Classification Metrics ++++
                  UAR: {self.test_met.uar:.4f}
                  WAR: {self.test_met.war:.4f}
                  macroF1: {self.test_met.macroF1:.4f}
                  microF1: {self.test_met.microF1:.4f}""")

class DownstreamMetricLearning(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hp = hparams
        self.dataset = CustomEmoDataset(self.hp.datadir, self.hp.labelpath, maxseqlen=self.hp.maxseqlen)
        if self.hp.pretrained_path is not None:
            self.model = PretrainedEmoFeatureHead.load_from_checkpoint(self.hp.pretrained_path, strict=False,
                                                                n_classes=self.dataset.nemos,
                                                                backend=self.hp.model_type)
        else:
            self.model = PretrainedEmoFeatureHead(n_classes=self.dataset.nemos,
                                           backend=self.hp.model_type)
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor(
            [counter[c] for c in self.dataset.emoset]
        ).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(
            f"Weigh losses by prior distribution of each class: {weights}."
        )
        
        if not hasattr(self.hp,'xent_alpha'):
            raise ValueError("Should pass xent_alpha for loss=xent_alpha*xent_loss+(1-xent_alpha)*metric_loss")
        self.xent_alpha = self.hp.xent_alpha
        assert self.xent_alpha >=0 and self.xent_alpha<=1, "xent_alpha should be in [0,1] interval"
        #TODO allow changing loss and miner type/hparams from args
        self.xent_criterion = nn.CrossEntropyLoss(weight=weights)
        self.metric_criterion = pml_losses.ContrastiveLoss(pos_margin=1, neg_margin=1)
        #TODO miner and allowed combinations
        # self.miner=None
        # self.use_miner=self.hp.use_miner

        
        with open(os.path.join(hparams.output_path,'predictions.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["file", "true", "predicted"])
            del writer
        
        # Define metrics
        if hasattr(self.dataset, 'val_dataset'):
            self.valid_met = ConfusionMetrics(self.dataset.nemos)
        if hasattr(self.dataset, 'test_dataset'):
            self.test_met = ConfusionMetrics(self.dataset.nemos)

    def forward(self, x, length):
        return self.model(x, length)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.trainable_params(), lr=self.hp.lr)
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(dataset=self.dataset.train_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=True,
                            num_workers=self.hp.nworkers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if not hasattr(self.dataset, 'val_dataset'):
            return
        loader = DataLoader(dataset=self.dataset.val_dataset,
                            collate_fn=self.dataset.seqCollate,
                            batch_size=self.hp.batch_size,
                            shuffle=False,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(dataset=self.dataset.test_dataset,
                            batch_size=1,
                            num_workers=self.hp.nworkers,
                            drop_last=False)
        return loader

    def training_step(self, batch, batch_idx):
        feats, length, label, fname  = batch
        reps,pout = self(feats, length)
        xent_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        total_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        metric_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        if self.xent_alpha<1:
            metric_loss = self.metric_criterion(reps,label)
            total_loss += (1-self.xent_alpha)*metric_loss
        if self.xent_alpha>0:
            xent_loss = self.xent_criterion(pout, label)
            total_loss += self.xent_alpha*xent_loss
        tqdm_dict = {'xent_loss': xent_loss,'metric_loss':metric_loss,'total_train_loss':total_loss}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        feats, length, label, fname  = batch
        reps,pout = self(feats, length)
        xent_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        total_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        metric_loss=torch.tensor(0,dtype=torch.float32).to(device=reps.device)
        if self.xent_alpha<1:
            metric_loss = self.metric_criterion(reps,label)
            total_loss += (1-self.xent_alpha)*metric_loss
        if self.xent_alpha>0:
            xent_loss = self.xent_criterion(pout, label)
            total_loss += self.xent_alpha*xent_loss
        for l, p in zip(label, pout):
            self.valid_met.fit(int(l), int(p.argmax()))
        tqdm_dict = {'xent_loss': xent_loss,'metric_loss':metric_loss,'total_valid_loss':total_loss}
        self.log_dict(tqdm_dict, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        print (self.valid_met.uar)
        self.log('valid_UAR', self.valid_met.uar)
        self.log('valid_WAR', self.valid_met.war)
        self.log('valid_macroF1', self.valid_met.macroF1)
        self.log('valid_microF1', self.valid_met.microF1)
        self.valid_met.clear()

    def test_step(self, batch, batch_idx):
        feats, label, fname = batch
        length = torch.LongTensor([feats.size(1)]).to(label.device)
        _,pout = self(feats, length)
        prediction = int(pout.argmax())
        
        arr = []
        for i in range(len(label)):
            arr.append([fname[i], int(label[i]), prediction])

        with open(os.path.join(self.hp.output_path,'predictions.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(arr)
            del writer

        self.test_met.fit(int(label), prediction)

    def on_test_epoch_end(self):
        """Report metrics."""
        self.log('test_UAR', self.test_met.uar, logger=True)
        self.log('test_WAR', self.test_met.war, logger=True)
        self.log('test_macroF1', self.test_met.macroF1, logger=True)
        self.log('test_microF1', self.test_met.microF1, logger=True)

        print(f"""++++ Classification Metrics ++++
                  UAR: {self.test_met.uar:.4f}
                  WAR: {self.test_met.war:.4f}
                  macroF1: {self.test_met.macroF1:.4f}
                  microF1: {self.test_met.microF1:.4f}""")
