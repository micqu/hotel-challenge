import numpy as np
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import data_loader as dl
import time
import copy
import wandb
import utility
import yaml

USE_AMP = False # set to True to use NVIDIA apex 16-bit precision

def main():
    with open('sweep_config.yaml', 'r') as f:
        sweep_config = yaml.load(f, yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config, project="hotel-challenge")
    wandb.agent(sweep_id, train_model)
    
def train_model():
    # Init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a new wandb run
    wandb.init()
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    
    # Build dataset
    train_loader, valid_loader, label_enc, n_classes = dl.get_train_valid_loader(
        data_dir='data/train_images',
        meta_data_file='data/train.csv',
        batch_size=config.batch_size,
        augment=True,
        random_seed=0
    )
    
    # Make resnet
    model = utility.initialize_resnet(n_classes, config.resnet_type,
                                      config.use_feature_extract)
    model = model.to(device)
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = model.parameters()
    if config.use_feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    
    # Define criterion + optimizer
    criterion = nn.CrossEntropyLoss()
    if config.optimizer=='sgd':
        optimizer = optim.SGD(params_to_update, lr=config.learning_rate)
    elif config.optimizer=='rmsprop':
        optimizer = optim.RMSprop(params_to_update, lr=config.learning_rate)
    elif config.optimizer=='adam':
        optimizer = optim.Adam(params_to_update, lr=config.learning_rate)
    
    if USE_AMP:
        scaler = torch.cuda.amp.GradScaler()
    
    # Define scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=10, epochs=config.epochs,
                                              anneal_strategy=config.scheduler,
                                              steps_per_epoch=len(train_loader))
    
    # Run train loop
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_map = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.enable_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_map += utility.calculate_map(outputs, labels)
            
        model.eval()
        valid_loss = 0.0
        valid_map = 0.0
        for inputs, labels in valid_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * inputs.size(0)
            valid_map += utility.calculate_map(outputs, labels)

        train_loss /= len(train_loader.dataset)
        train_map /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        valid_map /= len(valid_loader.dataset)
        
        wandb.log({"train_loss": train_loss,
                   "train_map": train_map,
                   "epoch": epoch,
                   "valid_loss": valid_loss,
                   "valid_map": valid_map})
                
if __name__ == "__main__":
    main()