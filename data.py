import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
import utility
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
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

def build_dataset(batch_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomApply([utility.AddGaussianNoise(0., 1.)], p=0.5),
            transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }
    
    df = pd.read_csv('data/train.csv')
    df = df.groupby('hotel_id').filter(lambda x : len(x)>50) #remove hotel ids with only 1 sample
    df, label_encoder = utility.encode_labels(df)
    num_classes = len(df['label'].value_counts())
        
    y = df.label
    X = df.drop(['label', 'timestamp'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7,
                                                          stratify=y, random_state=0)
    
    df_train = X_train.merge(y_train, left_index=True, right_index=True)
    df_valid = X_valid.merge(y_valid, left_index=True, right_index=True)
    
    #df_train = df_train.iloc[:int(len(df_train)*0.2),:] #20% of train set
    #df_valid = df_valid.iloc[:int(len(df_valid)*0.1),:]
    
    data_dir = 'data/train_images'
    data_files = {'train': df_train, 'valid': df_valid}
    image_datasets = {x: HotelImagesDataset(data_files[x],
                                            root_dir=data_dir,
                                            transform=data_transforms[x])
                      for x in ['train', 'valid']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=NUM_WORKERS)
                   for x in ['train', 'valid']}
    
    return dataloaders, label_encoder, num_classes