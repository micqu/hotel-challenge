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

PRINT_STATUS = True
BATCH_SIZE = 8
EPOCHS = 1
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
    
    for image_size in [64, 128, 224]:
        for chain_id in df['chain']:
            chain_df = df.loc[df['chain'] == chain_id]
        
            # Build dataset
            train_loader, valid_loader = dl.get_train_valid_loader(chain_df,
                                                                data_dir='data/train_images',
                                                                batch_size=BATCH_SIZE,
                                                                image_size=image_size,
                                                                augment=True,
                                                                random_seed=0)

            model = utility.initialize_resnet(num_classes, "resnet18",
                                              feature_extract=True, use_pretrained=True)
            
            filename = f"models/chain_{chain_id}_model.pt"
            if path.exists(filename):
                model.load_state_dict(torch.load(filename))
            
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                max_lr=10, epochs=EPOCHS,
                                                anneal_strategy=ANNEAL_STRAT,
                                                steps_per_epoch=len(train_loader))

            print(f"Now training chain {chain_id} for size {image_size}")
            trained_model = trainer.train_model(device=device,
                                                model=model,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                train_loader=train_loader,
                                                valid_loader=valid_loader,
                                                scheduler=scheduler,
                                                epochs=EPOCHS,
                                                print_status=PRINT_STATUS)
            
            torch.save(trained_model.state_dict(), filename)
            print(f"Done training {chain_id} for size {image_size}")
                
if __name__ == "__main__":
    main()