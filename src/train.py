import os

from datetime import datetime
import hydra

import torch
from hydra.utils import instantiate
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from base import Experiment
from src.utils.callbacks.loss_logger import LossLogger
from src.utils.callbacks.metric_logger import MetricLogger


SEED=42
pl.seed_everything(SEED)
seed_everything(SEED, workers=True)

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)
    cfg.update({"day": datetime.now().strftime("%Y-%m-%d")})
    cfg.update({"log_dir": f'{os.environ["PROJECT_ROOT"]}/logs/{cfg.dataset.name}/x{cfg.sampling}/{cfg.day}/{cfg.model.name}/'})

    data_loader = instantiate(cfg.dataset.datamodule)
    experiment = Experiment(cfg)
    logger = TensorBoardLogger(
        f'{os.environ["PROJECT_ROOT"]}/logs/{cfg.dataset.name}/x{cfg.sampling}/{cfg.day}/{cfg.model.name}/',
            name=cfg.nickname)
    cfg.update({"log_dir": logger.log_dir})

    default_callbacks = [
                            LossLogger(),
                            MetricLogger(test_subsets=["validation", "test"], validation_subsets=["validation"]),
                            RichModelSummary(max_depth=4),
                            instantiate(cfg.checkpoint),
                         ]

    callback_list = instantiate(cfg.model.callbacks) + default_callbacks if hasattr(cfg.model, 'callbacks') else default_callbacks

    trainer = Trainer(max_epochs=cfg.model.train.max_epochs, logger=logger,
                      devices=cfg.devices,
                      callbacks=callback_list)

    trainer.fit(experiment, datamodule=data_loader)
    data_loader = instantiate(cfg.dataset.datamodule, batch_size=1)
    print(f'Best model path: {cfg.log_dir}/checkpoints/best.ckpt')
    ckpt = torch.load(f'{cfg.log_dir}/checkpoints/best.ckpt', map_location=f'cuda:{cfg.devices[0]}', weights_only=False)
    experiment.load_state_dict(ckpt['state_dict'])
    trainer.test(experiment, datamodule=data_loader)

    return 0

if __name__ == '__main__':
    train()
