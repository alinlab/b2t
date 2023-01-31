import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    """
    CUB dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(self, root, split, transform, pseudo_bias, metadata_csv_name="metadata.csv"):
        self.root = root
        self.transform = transform
        
        if not os.path.exists(self.root):
            raise ValueError(
                f"{self.root} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        print(f"Reading '{os.path.join(self.root, metadata_csv_name)}'")
        self.metadata_df = pd.read_csv(os.path.join(self.root, metadata_csv_name))

        # Get the y values
        self.y_array = self.metadata_df["y"].values

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        
        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2,}
        
        # split
        assert split in ("train", "val", "test"), f"{split} is not a valid split"
        mask = self.split_array == self.split_dict[split]

        indices = np.where(mask)[0]
        
        self.filename = self.filename_array[indices]
        self.targets = self.y_array[indices]
        self.biases = self.confounder_array[indices]
        
        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias).numpy()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        X = Image.open(os.path.join(self.root, self.filename[index])).convert("RGB")
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
