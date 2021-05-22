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
Image.MAX_IMAGE_PIXELS = None

PRINT_STATUS = False # set to False to avoid print messages
USE_AMP = False # set to True to use NVIDIA apex 16-bit precision
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
ANNEAL_STRAT = "cos"

def main():
    # Init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the meta data file
    df = pd.read_csv('data/train.csv',)
    df = df.drop(['timestamp'], axis=1)
    df, label_encoder = utility.encode_labels(df)
    num_classes = len(df['label'].value_counts())
    
    for chain_id in df['chain']:
        chain_df = df.loc[df['chain'] == chain_id]
    
        # Build dataset
        train_loader, valid_loader = dl.get_train_valid_loader(chain_df,
                                                            data_dir='data/train_images',
                                                            batch_size=BATCH_SIZE,
                                                            augment=True,
                                                            random_seed=0)

        model = utility.initialize_resnet(num_classes, "resnet50",
                                          feature_extract=True, use_pretrained=True)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=10, epochs=EPOCHS,
                                              anneal_strategy=ANNEAL_STRAT,
                                              steps_per_epoch=len(train_loader))

        trainer.train_model(device=device,
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            train_loader=train_loader,
                            valid_loader=valid_loader,
                            scheduler=scheduler,
                            epochs=EPOCHS)
        print("Done training")

def print_status_bar(epoch, total_epoch, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    print(f"At epoch: {epoch}/{total_epoch}" +
        f"- train loss: {train_loss:.4f} - " + f"train map: {train_map:.4f}" +
        f"- valid loss: {valid_loss:.4f} - " + f"valid map: {valid_map:.4f}" +
        "- time spent: " + f"{time_taken:.2f}" + "\n")
                
if __name__ == "__main__":
    main()