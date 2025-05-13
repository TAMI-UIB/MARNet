import os

from typing import Any

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils.metrics_calculator import MetricCalculator
from src.utils.metrics_calculator import NonRefMetricCalculator


class MetricLogger(Callback):
    def __init__(self, metric_monitor='PSNR', test_subsets=['validation', 'test'], validation_subsets=['validation'], metric_list=['PSNR', 'SSIM', 'SAM', 'ERGAS'], path=None) -> None:
        super(MetricLogger, self).__init__()
        self.metric_list = metric_list
        self.metric_monitor = metric_monitor
        self.validation_subsets = validation_subsets
        self.test_subsets = test_subsets
        self.subsets = ['train']+test_subsets
        self.fit_subsets = ['train']+validation_subsets
        self.metrics = {k: MetricCalculator(metric_list) for k in self.subsets}
        self.path = path

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.metrics['train'].clean()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in self.validation_subsets:
            self.metrics[subset].clean()

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in self.test_subsets:
            self.metrics[subset].clean()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        writer = trainer.logger.experiment
        epoch = trainer.current_epoch
        metric_means = {subset: self.metrics[subset].get_means() for subset in self.fit_subsets}
        for subset in self.fit_subsets:
            for metric in self.metric_list:
                writer.add_scalar(f"{metric}/{subset}", metric_means[subset][metric], epoch)
        for metric in self.metric_list:
            writer.add_scalars(f"{metric}/comparison", {k: v[metric] for k, v in metric_means.items()}, epoch)
        pl_module.log(f'{self.metric_monitor}', metric_means['validation'][self.metric_monitor],  prog_bar=True)


    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        _, gt, _ = batch
        self.metrics['train'].update(preds=outputs['output']['pred'], targets=gt['gt'])

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        _, gt, _ = batch
        subset = self.validation_subsets[dataloader_idx]
        self.metrics[subset].update(preds=outputs['pred'], targets=gt['gt'])

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        _, gt, _ = batch
        subset = self.test_subsets[dataloader_idx]
        self.metrics[subset].update(preds=outputs['pred'], targets=gt['gt'])

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        path = os.path.join(pl_module.cfg.log_dir, 'metric_report') if self.path is None else self.path
        os.makedirs(path, exist_ok=True)
        base_data = {
            "day": str(pl_module.cfg.day),
            "models": pl_module.cfg.model.name,
            "nickname": pl_module.cfg.nickname,
            "log_dir": path,
        }
        for subset in self.test_subsets:
            means = self.metrics[subset].get_means()
            mean_data = pd.DataFrame({**base_data, **means},
                                     index=[0])
            mean_file_path = f'{path}/{subset}_metrics.csv'

            if self.path is None:
                mean_data.to_csv(mean_file_path, index=False)
            else:
                self.save_or_append_csv(mean_file_path, mean_data)

    @staticmethod
    def save_or_append_csv(file_path, new_data):
        try:
            old_data = pd.read_csv(file_path)
            pd.concat([old_data, new_data]).to_csv(file_path, index=False)
        except FileNotFoundError:
            new_data.to_csv(file_path, index=False)

class NonRefMetricLogger(Callback):
    def __init__(self, metric_list=['QNR','HQNR'], path=None) -> None:
        super(NonRefMetricLogger, self).__init__()
        self.metric_list = metric_list
        self.test_subsets = ['full_res']
        self.metrics = {k: NonRefMetricCalculator(metric_list) for k in self.test_subsets}
        self.path = path


    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in self.test_subsets:
            self.metrics[subset].clean()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_id: int, dataloader_idx=0) -> None:
        low, _, _ = batch
        subset = self.test_subsets[dataloader_idx]
        self.metrics[subset].update(pred=outputs['pred'], hs=low["hs"], pan=low['pan'])

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        path = os.path.join(pl_module.cfg.log_dir, 'metric_report') if self.path is None else self.path
        os.makedirs(path, exist_ok=True)

        base_data = {
            "day": str(pl_module.cfg.day),
            "models": pl_module.cfg.model.name,
            "nickname": pl_module.cfg.nickname,
            "log_dir": path,
        }
        for subset in self.test_subsets:

            metrics = self.metrics[subset].get_means()
            mean_data = pd.DataFrame({**base_data, **metrics},
                                     index=[0])

            metrics_file_path = f'{path}/{subset}_metrics.csv'
            if self.path is None:
                mean_data.to_csv(metrics_file_path, index=False)
            else:
                self.save_or_append_csv(metrics_file_path, mean_data)


    @staticmethod
    def save_or_append_csv(file_path, new_data):
        try:
            old_data = pd.read_csv(file_path)
            pd.concat([old_data, new_data]).to_csv(file_path, index=False)
        except FileNotFoundError:
            new_data.to_csv(file_path, index=False)