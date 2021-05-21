import os
import torch
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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