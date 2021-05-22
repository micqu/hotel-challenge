import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
import utility
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

IMAGE_SIZE = 64
NUM_WORKERS = 8

class HotelImagesDataset(Dataset):
    """Hotel images dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.classes = list(self.df['label'].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        chain_id = self.df.iloc[idx, self.df.columns.get_loc('chain')]
        img_name = os.path.join(self.root_dir,
                                str(chain_id),
                                self.df.iloc[idx, self.df.columns.get_loc('image')])

        image = io.imread(img_name)
        pil_image = Image.fromarray(image)
        y = self.df.iloc[idx, self.df.columns.get_loc('label')]
        if self.transform:
            X = self.transform(pil_image)

        return X, y

def get_train_valid_loader(df,
                           data_dir,
                           batch_size: int,
                           augment: bool,
                           random_seed: int,
                           train_size: float = 0.7,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    # define transforms
    valid_transform = transforms.Compose([
            utility.AddPadding(),
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            utility.AddPadding(),
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomApply([utility.AddGaussianNoise(0., 1.)], p=0.5),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
    
    # Load datasets
    train_dataset = HotelImagesDataset(df, root_dir=data_dir, transform=train_transform)
    valid_dataset = HotelImagesDataset(df, root_dir=data_dir, transform=valid_transform)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor((1 - train_size) * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers,
                                               pin_memory=pin_memory)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers,
                                               pin_memory=pin_memory,
    )
    
    return train_loader, valid_loader

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True):
    #TODO: Define a test loader based on test data, if it is on Kaggle?
    pass

def get_dataset(df, data_dir, transform):
    dataset = HotelImagesDataset(df, root_dir=data_dir, transform=transform)
    return dataset
    