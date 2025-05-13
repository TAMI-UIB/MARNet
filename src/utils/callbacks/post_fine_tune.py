import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim import Adam


class PostFineTune(Callback):
    def __init__(self, max_epochs, loss_function, lr) -> None:
        super(PostFineTune, self).__init__()
        self.epoch = max(max_epochs - 100, 1)
        self.loss_function = loss_function
        self.lr =lr

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_epoch = trainer.current_epoch
        if self.epoch == current_epoch:
            cfg = pl_module.cfg
            log_path = f"{cfg.log_dir}/checkpoints/best.ckpt"
            ckpt = torch.load(log_path, weights_only=False)
            pl_module.load_state_dict(ckpt["state_dict"])
            pl_module.model.fine_tune()
            pl_module.loss_criterion = self.loss_function
            pl_module.optimizers =  [Adam(pl_module.model.parameters(), lr=self.lr)]
            trainable_params = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in pl_module.model.parameters())
            print(f"Number of parameters trainable parameters: {trainable_params}/{total_params}")
