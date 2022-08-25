from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from vae_pytorch.vanilla_vae import Vanilla_VAE
from pytorch_lightning import Trainer
from experiment import VAE_Experiment
from dataset import VAEDataset
from pathlib import Path
import argparse
import yaml
import os


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument(
    '--config',
    '-c',
    dest="filename",
    metavar='FILE',
    help='path to the config file',
    default='configs/vae.yaml'
)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['name']
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = Vanilla_VAE(**config['model_params'])
experiment = VAE_Experiment(model, config['exp_params'])

data = VAEDataset(**config['data_params'], pin_memory=len(config['trainer_params']['devices']) != 0)

data.setup()
runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir , 'checkpoints'),
            monitor='val_loss',
            save_last=True
        ),
    ],
    strategy=DDPStrategy(find_unused_parameters=False),
    **config['trainer_params']
)

Path(f'{tb_logger.log_dir}/Samples').mkdir(exist_ok=True, parents=True)
Path(f'{tb_logger.log_dir}/Reconstructions').mkdir(exist_ok=True, parents=True)

print(f'======= Training {config["name"]} =======')
runner.fit(experiment, datamodule=data)
