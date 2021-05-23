import numpy as np
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pandas as pd
import data_loader as dl
import utility
import trainer
from PIL import Image
from os import path
Image.MAX_IMAGE_PIXELS = None
from scipy.io import savemat

def main():
    df = pd.read_csv('data/train.csv')
    df, _ = utility.encode_labels(df)
    data_loader = dl.get_full_data_loader(df, data_dir='data/train_images',
                                          batch_size=128,
                                          image_size=32)
    train_dataset_arr = next(iter(data_loader))[0].numpy()
    
    zca = utility.ZCA()
    zca.fit(train_dataset_arr)
    zca_mat = zca.ZCA_mat
    zca_mean = zca.mean
    zca_dic = {"zca_matrix": zca_mat, "zca_mean": zca_mean}
    savemat("data/zca_data.mat", zca_dic)
                    
if __name__ == "__main__":
    main()