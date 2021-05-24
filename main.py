import numpy as np
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pandas as pd
import data_loader as dl
import time
import copy
import utility
import yaml
import trainer
from PIL import Image
from os import path
Image.MAX_IMAGE_PIXELS = None
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from torchvision import transforms

BATCH_SIZE = 128
EPOCHS = 100
LR = 0.01
ANNEAL_STRAT = "cos"
IMAGE_SIZE = 64
FEATURE_EXTRACT = True
APPLY_ZCA_TRANS = True
DATA_DIR = 'data/train_images'
NETS = ['resnext'] # train on resnext, no pretraining

def main():
    # Init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the meta data file
    df = pd.read_csv('./data/train.csv')
    df, label_encoder = utility.encode_labels(df)
    num_classes = len(df['label'].value_counts())
    
    # Generate the ZCA matrix if enabled
    if APPLY_ZCA_TRANS:
        print("Making ZCA matrix ...")
        data_loader = dl.get_full_data_loader(df, data_dir=DATA_DIR,
                                              batch_size=BATCH_SIZE,
                                              image_size=IMAGE_SIZE)
        train_dataset_arr = next(iter(data_loader))[0].numpy()
        zca = utility.ZCA()
        zca.fit(train_dataset_arr)
        zca_dic = {"zca_matrix": zca.ZCA_mat, "zca_mean": zca.mean}
        savemat("./data/zca_data.mat", zca_dic)
        print("Completed making ZCA matrix")
    
    # Define normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    # Define transforms
    train_transform = transforms.Compose([
            utility.AddPadding(),
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            normalize,
            #transforms.RandomApply([utility.AddGaussianNoise(0., 1.)], p=0.5)
        ])
    valid_transform = transforms.Compose([
            utility.AddPadding(),
            transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
    ])
    
    # Create a train and valid dataset
    train_dataset = dl.HotelImagesDataset(df, root_dir=DATA_DIR,
                                          transform=train_transform)
    valid_dataset = dl.HotelImagesDataset(df, root_dir=DATA_DIR,
                                          transform=valid_transform)
            
    # Get a train and valid data loader
    train_loader, valid_loader = dl.get_train_valid_loader(train_dataset,
                                                           valid_dataset,
                                                           batch_size=BATCH_SIZE,
                                                           random_seed=0)
        
    for net_type in NETS: # train for every net
        model = utility.initialize_net(num_classes, net_type,
                                       feature_extract=FEATURE_EXTRACT)
    
        # Gather the parameters to be optimized/updated in this run.
        params_to_update = utility.get_model_params_to_train(model, FEATURE_EXTRACT)
    
        # Send model to GPU
        model = model.to(device)

        # Make criterion
        criterion = nn.CrossEntropyLoss()
        
        # Make optimizer + scheduler
        optimizer = optim.Adam(params_to_update, lr=LR)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                max_lr=10, epochs=EPOCHS,
                                                anneal_strategy=ANNEAL_STRAT,
                                                steps_per_epoch=len(train_loader))

        trained_model = trainer.train_model(device=device,
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            train_loader=train_loader,
                                            valid_loader=valid_loader,
                                            net_type=net_type,
                                            scheduler=scheduler,
                                            epochs=EPOCHS,
                                            apply_zca_trans=APPLY_ZCA_TRANS)
    
        utility.save_current_model(trained_model, f"./models/model_{net_type}.pt")
                    
if __name__ == "__main__":
    main()