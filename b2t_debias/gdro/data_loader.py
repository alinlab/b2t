import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from PIL import Image

from celeba_dataset import CustomCelebA
from cub_dataset import CUBDataset


def prepare_data(args):
    if args.dataset == 'celeba':
        return prepare_celeba(args)
    elif args.dataset == 'cub':
        return prepare_cub(args)
    else:
        raise NotImplementedError

    
def prepare_cub(args):
    transform_train = transforms.Compose([
                        transforms.Resize((int(args.image_size * 256.0 / 224.0), 
                                           int(args.image_size * 256.0 / 224.0))),
                        transforms.RandomCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_valid = transforms.Compose([
                        transforms.Resize((int(args.image_size * 256.0 / 224.0), 
                                           int(args.image_size * 256.0 / 224.0))),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cub_train = CUBDataset(
                    root='/data/waterbird_complete95_forest2water2',
                    split='train', 
                    transform=transform_train,
                    pseudo_bias=args.pseudo_bias)
    cub_valid = CUBDataset(
                    root='/data/waterbird_complete95_forest2water2',
                    split='val', 
                    transform=transform_valid,
                    pseudo_bias=None)
    cub_test = CUBDataset(
                    root='/data/waterbird_complete95_forest2water2',
                    split='test', 
                    transform=transform_valid,
                    pseudo_bias=None)

    labeled_indices = np.arange(len(cub_train))
    train_labeled_dataset = cub_train
    train_unlabeled_dataset = cub_train
    valid_dataset = cub_valid
    test_dataset = cub_test

    group = np.zeros(len(labeled_indices)).astype('int')
    group[np.where(cub_train.targets == 1)[0]] += 2
    group[np.where(cub_train.biases == 1)[0]] += 1

    group_sample_count = np.zeros(4)
    weight = np.zeros(4)
    for g in np.unique(group):
        group_sample_count[g] = len(np.where(group == g)[0])
        weight[g] = 1. / group_sample_count[g]
    samples_weight = np.array([weight[g] for g in group])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      num_workers=args.num_workers,
                                      drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader


def prepare_celeba(args):
    """Create and return Dataloader."""
    transform_train = transforms.Compose([
                        transforms.Resize(int(args.image_size * 256.0 / 224.0)),
                        transforms.RandomCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_valid = transforms.Compose([
                        transforms.Resize(int(args.image_size * 256.0 / 224.0)),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # dataset = ImageFolder(image_path, transform)
    celeba_train = CustomCelebA(
                        root=args.data_root,
                        split='train', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_train, 
                        pseudo_bias=args.pseudo_bias)
    celeba_valid = CustomCelebA(
                        root=args.data_root,
                        split='valid', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_valid)
    celeba_test = CustomCelebA(
                        root=args.data_root,
                        split='test', 
                        target_attr=args.target_attr, 
                        bias_attr=args.bias_attr,
                        transform=transform_valid)

    labeled_indices = np.arange(len(celeba_train))
    train_labeled_dataset = celeba_train
    train_unlabeled_dataset = celeba_train

    group = celeba_valid.targets * 2 + celeba_valid.biases
    valid_indices = list()
    for i in range(4):
        valid_indices.append(np.where(group == i)[0][:int(len(np.where(group == i)[0])*args.val_frac)])
    valid_indices = np.concatenate(valid_indices)
    valid_indices.sort()
    valid_dataset = Subset(celeba_valid, valid_indices)

    test_dataset = celeba_test

    group = np.zeros(len(labeled_indices)).astype('int')
    group[np.where(celeba_train.targets == 1)[0]] += 2
    group[np.where(celeba_train.biases == 1)[0]] += 1
    group_sample_count = np.zeros(4)
    weight = np.zeros(4)
    for g in np.unique(group):
        group_sample_count[g] = len(np.where(group == g)[0])
        weight[g] = 1. / group_sample_count[g]
    samples_weight = np.array([weight[g] for g in group])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_labeled_loader = DataLoader(dataset=train_labeled_dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      num_workers=args.num_workers,
                                      drop_last=True)
        
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader