import os
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from hydra.utils import instantiate
import rootutils



rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.base import Experiment
from src.utils.callbacks.metric_logger import NonRefMetricLogger




@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def test_full_res(cfg: DictConfig):

    cfg.model.train.batch_size = 1
    datamodule = instantiate(cfg.dataset.datamodule)

    weights = torch.load(cfg.model.ckpt_path, map_location=f'cuda:{cfg.devices[0]}', weights_only=False)


    cfg.model.module = weights['cfg'].model.module
    experiment = Experiment(cfg)

    experiment.load_state_dict(weights['state_dict'])

    log_dir = f"{os.environ['PROJECT_ROOT']}/logs/results_{cfg.dataset.name}"
    non_ref_metrics = ['QNR', 'HQNR']

    datamodule.setup(stage='full_res')

    callback_list = [NonRefMetricLogger(metric_list=non_ref_metrics, path=log_dir)]

    trainer = Trainer(devices=cfg.devices, deterministic=True, callbacks=callback_list, logger=False)
    trainer.test(experiment, datamodule=datamodule)

    return 0


if __name__ == '__main__':
    test_full_res()



