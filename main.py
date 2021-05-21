import numpy as np
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import time
import copy
import ml_metrics
import wandb
import utility
import data
import yaml

NUM_WORKERS = 8
PRINT_STATUS = False # set to False to avoid print messages
USE_AMP = False # set to True to use NVIDIA apex 16-bit precision

def main():
    with open('sweep_config.yaml', 'r') as f:
        sweep_config = yaml.load(f, yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config, project="hotel-challenge")
    wandb.agent(sweep_id, train_model)
    
def train_model():
    config_defaults = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'use_feature_extract': True,
        'resnet_type': 'resnet18'
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    
    # Build dataset
    dataloaders, label_encoder, num_classes = data.build_dataset(config.batch_size)
    
    # Make resnet
    model = utility.initialize_resnet(num_classes, config.resnet_type,
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
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=100, epochs=config.epochs,
                                              anneal_strategy=config.scheduler,
                                              steps_per_epoch=len(dataloaders['train']))
    
    # Run train loop
    start_time = time.time()
    num_steps = len(dataloaders['train'].dataset) // config.batch_size
    for epoch in range(1, config.epochs + 1):
        
        model.train()
        train_loss = 0.0
        train_map = 0.0
        for inputs, labels in dataloaders['train']:
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
            train_map += calculate_map(outputs, labels)
            
        model.eval()
        valid_loss = 0.0
        valid_map = 0.0
        for inputs, labels in dataloaders['valid']:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * inputs.size(0)
            valid_map += calculate_map(outputs, labels)

        train_loss /= len(dataloaders['train'].dataset)
        train_map /= len(dataloaders['train'].dataset)
        valid_loss /= len(dataloaders['valid'].dataset)
        valid_map /= len(dataloaders['valid'].dataset)
        
        time_taken = time.time() - start_time
        wandb.log({"train_loss": train_loss,
                   "train_map": train_map,
                   "epoch": epoch,
                   "time_taken": time_taken,
                   "valid_loss": valid_loss,
                   "valid_map": valid_map})
        
        if PRINT_STATUS:
            print_status_bar(epoch,
                             config.epochs,
                             train_loss,
                             train_map,
                             valid_loss,
                             valid_map,
                             time_taken)
        
def print_status_bar(epoch, total_epoch, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    print(f"At epoch: {epoch}/{total_epoch}" +
        f"- train loss: {train_loss:.4f} - " + f"train map: {train_map:.4f}" +
        f"- valid loss: {valid_loss:.4f} - " + f"valid map: {valid_map:.4f}" +
        "- time spent: " + f"{time_taken:.2f}" + "\n")
          
def calculate_map(outputs, labels):
    top_k = torch.topk(outputs, 5)
    preds = top_k.indices.detach().cpu().numpy()
    corrects = labels.detach().cpu().numpy()
    return ml_metrics.mapk([[x] for x in corrects], preds, k=5)
                
if __name__ == "__main__":
    main()