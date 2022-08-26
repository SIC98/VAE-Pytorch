import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import tensor
import numpy as np
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
            M_N=self.params['kld_weight'], # 1.0
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
    
    def on_validation_end(self) -> None:
        self.sample_images()
        self.distribution_of_latent_variable()

    def distribution_of_latent_variable(self):
        N = 10
        plt.figure(figsize=(8, 6))
        merged_z = []
        merged_label = []

        for test_input, test_label in self.trainer.datamodule.test_dataloader():

            test_input = test_input.to(self.curr_device)
            z = self.model.get_latent_variable(test_input)
            merged_z.append(z)
            merged_label.append(test_label)

        merged_z = torch.cat(merged_z, 0).cpu()
        merged_label = torch.cat(merged_label, 0).cpu()

        plt.scatter(merged_z[:, 0], merged_z[:, 1], c=merged_label, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        self.z_range = 4
        axes.set_xlim([-self.z_range, self.z_range])
        axes.set_ylim([-self.z_range, self.z_range])
        plt.grid(True)
        plt.savefig(
            os.path.join(
                self.logger.log_dir,
                'Distribution',
                f'{self.logger.name}_Epoch_{self.current_epoch}.png'
            )
        )
        plt.close()
    
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


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    '''Create an N-bin discrete colormap from the specified input map'''

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
