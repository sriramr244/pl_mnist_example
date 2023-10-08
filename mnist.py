import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.data_dir = kwargs.get('data_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory')
        self.val_ratio = kwargs.get('val_ratio')

        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((self.val_ratio >= 0) and (self.val_ratio <= 1)), error_msg
        # Data: data transformation strategy
        self.do_transform = transforms.Compose([transforms.RandomAffine(15, (0.1, 0.1), (0.95, 1.05)), transforms.ToTensor()])
        self.no_transform = transforms.Compose([transforms.ToTensor()])

        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, transform=self.do_transform, download=True)
        self.dataset_val = datasets.MNIST(root=self.data_dir, train=True, transform=self.no_transform, download=True)

        num_train = len(self.dataset_train)
        indices = list(range(num_train))
        split = int(np.floor(self.val_ratio * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.dataset_tr_indices = SubsetRandomSampler(train_idx)
        self.dataset_val_indices = SubsetRandomSampler(valid_idx)

        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, transform=self.no_transform, download=True)


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=self.dataset_tr_indices, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, sampler=self.dataset_val_indices, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
