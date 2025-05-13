
import os

import hydra
import torch

from omegaconf import DictConfig
from pytorch_lightning import Trainer
import rootutils



rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.base import Experiment

from src.utils.callbacks.metric_logger import MetricLogger

from hydra.utils import instantiate


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def test(cfg: DictConfig):

    cfg.model.train.batch_size = 1
    datamodule = instantiate(cfg.dataset.datamodule)

    weights = torch.load(cfg.model.ckpt_path, map_location=f'cuda:0', weights_only=False)
    cfg.model = weights['cfg'].model
    experiment = Experiment(cfg)
    experiment.load_state_dict(weights['state_dict'])

    log_dir = f"{os.environ['PROJECT_ROOT']}/logs/results_{cfg.dataset.name}"
    metrics = ['PSNR', 'SSIM','SAM', 'ERGAS','Q2N']

    datamodule.setup(stage='test')

    callback_list = [MetricLogger(metric_list=metrics, path=log_dir)]

    trainer = Trainer(devices=cfg.devices, deterministic=True, callbacks=callback_list, logger=False)
    trainer.test(experiment, datamodule=datamodule)

    return 0
if __name__ == '__main__':
    test()



