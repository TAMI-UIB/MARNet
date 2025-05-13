import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LossLogger(Callback):
    def __init__(self,) -> None:
        super(LossLogger, self).__init__()

        self.subsets = ['train', 'validation']

    def on_train_epoch_start(self, trainer: pl.Trainer,
                                   pl_module: pl.LightningModule):
        for subset in self.subsets:
            pl_module.loss[subset] = 0


    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                      pl_module: pl.LightningModule):

        writer = trainer.logger.experiment
        epoch = trainer.current_epoch
        writer.add_scalars(f"loss/comparison", {k: v for k, v in pl_module.loss.items() }, epoch)
        for subset in pl_module.fit_subsets:
            writer.add_scalar(f"loss/{subset}", pl_module.loss[subset], epoch)
