import torchvision.utils as vutils
import pytorch_lightning as pl
from torch import tensor
import torch
import os


class VAE_Experiment(pl.LightningModule):
    def __init__(self, model, params: dict) -> None:
        super().__init__()
        self.model = model
        self.params = params
    
    def forward(self, input: tensor):
        return self.model(input)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, _labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        real_img, _labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
    
    def on_validation_end(self) -> None:
        self.sample_images()
    
    def sample_images(self):
        test_input, _test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        
        recons = self.model.generate(test_input)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                'Reconstructions',
                f'recons_{self.logger.name}_Epoch_{self.current_epoch}.png'
            ),
            normalize=True,
            nrow=12
        )

        samples = self.model.sample(
            144,
            self.curr_device
        )

        vutils.save_image(
            samples.cpu().data,
            os.path.join(
                self.logger.log_dir,
                'Samples',
                f'{self.logger.name}_Epoch_{self.current_epoch}.png'
            ),
            normalize=True,
            nrow=12
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.params['scheduler_gamma']
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
