import os
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


class Experiment(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # Experiment configuration
        self.cfg = cfg
        # Define subsets
        self.subsets = ['train', 'validation', 'test']
        self.fit_subsets = ['train', 'validation']
        # Define models and loss
        self.model = instantiate(cfg.model.module)

        self.loss_criterion = instantiate(cfg.model.train.loss)
        # Number of models parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Exact number of parameters: {self.num_params}")
        # Loss report
        self.loss = {subset: 0 for subset in self.fit_subsets}

    def forward(self, batch):
        low, gt, names = batch
        pan = low['pan'].to(self.device)
        hs = low['hs'].to(self.device)
        output = self.model(pan=pan, hs=hs)
        return output

    def training_step(self, batch, idx):
        low, gt, name = batch
        output = self.forward(batch)
        loss = self.loss_criterion(output, gt)
        self.loss_report(loss[0].item(), 'train')
        return {"loss": loss[0], "output": output}

    def validation_step(self, batch, idx, dataloader_idx=0):
        low, gt, name = batch
        output = self.forward(batch)
        loss = self.loss_criterion(output, gt)
        self.loss_report(loss[0].item(), 'validation')
        return output

    def test_step(self, batch, idx, dataloader_idx=0):
        output = self.forward(batch)
        return output

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.train.optimizer,params=self.parameters())
        return {'optimizer': optimizer}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['cfg'] = self.cfg
        checkpoint['current_epoch'] = self.current_epoch

    def loss_report(self, loss, subset):
        self.loss[subset] += loss