"""
CelebA Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CelebA(Dataset):
    def __init__(self, data_dir='/data/celeba', split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'list_attr_celeba.csv'), delim_whitespace=True)
        self.split_df = pd.read_csv(os.path.join(self.data_dir, 'list_eval_partition.csv'), delim_whitespace=True)
        self.metadata_df['partition'] = self.split_df['partition']
        self.metadata_df = self.metadata_df[self.split_df['partition'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df['Blond_Hair'].values
        self.confounder_array = self.metadata_df['Male'].values
        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['image_id'].values
        self.split_array = self.metadata_df['partition'].values

        self.targets = torch.tensor(self.y_array)
        self.targets_group = torch.tensor(self.group_array)
        self.targets_spurious = torch.tensor(self.confounder_array)

        self.transform = transform

        self.n_classes = 2
        self.n_groups = 4

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.data_dir, 'img_align_celeba', self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        x = self.transform(img)

        y = self.targets[idx]
        y_group = self.targets_group[idx]
        y_spurious = self.targets_spurious[idx]

        return x, (y, y_group, y_spurious), idx

def get_transform_celeba():
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_waterbirds(root_dir, bs_train=128, bs_val=128, num_workers=8):
    """
    Default dataloader setup for CelebA

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = CelebA(root_dir, split='train')
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)

    val_set = CelebA(root_dir, split='val')
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    test_set = CelebA(root_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
