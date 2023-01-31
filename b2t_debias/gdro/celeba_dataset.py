import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image


class CustomCelebA(datasets.CelebA):
    def __init__(self, root, split, target_attr, bias_attr, transform, pseudo_bias=None):
        super(CustomCelebA, self).__init__(root, split, transform=transform, download=True)
        
        self.targets = self.attr[:, target_attr]
        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias)
        else:
            self.biases = self.attr[:, bias_attr]
        
    def __getitem__(self, index):
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        y = self.targets[index]
        a = self.biases[index]
        
        if self.transform is not None:
            X = self.transform(X)
            
        ret_obj = {'x': X,
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   'filename': self.filename[index],
                   }

        return ret_obj
