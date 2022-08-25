import torch
from torch import nn
from torch import tensor
from torch.nn import functional as F

class Vanilla_VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: list = [32, 64, 128, 256, 512]) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder Layers
        encoder_channels = in_channels
        modules = []

        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        encoder_channels, out_channels=dim,
                        kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()
                )
            )
            encoder_channels = dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Decoder Layers
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def encode(self, input: tensor) -> tensor:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        var = self.fc_var(result)

        return [mu, var]

    def decode(self, z: tensor) -> tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: tensor, var: tensor) -> tensor:
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, var = self.encode(input)
        z = self.reparameterize(mu, var)
        return [self.decode(z), input, mu, var]
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
    
    def sample(self, num_samples:int, current_device: int):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: tensor) -> tensor:

        return self.forward(x)[0]
