from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, MNIST
from torchvision import transforms


class MyCelebA(CelebA):

    def _check_integrity(self) -> bool:
        return True


class VAEDataset(LightningDataModule):
    '''
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    '''

    def __init__(
        self,
        data_name: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        self.data_name = data_name
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        if self.data_name == 'celeba':

            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor()
            ])

            val_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor()
            ])

            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=False,
            )

            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=False,
            )

        elif self.data_name == 'mnist':

            train_transforms = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.ToTensor()
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.ToTensor()
            ])

            self.train_dataset = MNIST(
                self.data_dir,
                train=True,
                transform=train_transforms,
                download=True,
            )

            self.val_dataset = MNIST(
                self.data_dir,
                train=False,
                transform=val_transforms,
                download=True,
            )

        else:
            raise ValueError('Wrong data_name')


    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
