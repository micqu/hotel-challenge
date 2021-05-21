import numpy as np
from torch import nn
import torch.optim as optim
import torch
import torchvision
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

NUM_WORKERS = 8

PRINT_STATUS = False # set to False to avoid print messages
USE_AMP = False # set to True to use NVIDIA apex 16-bit precision

def main():
    sweep_config = {
        'method': 'bayes', #grid, random, bayes
        'metric': {
        'name': 'loss',
        'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
        },
        'parameters': {
            'epochs': {
                'values': [10, 20, 50]
            },
            'batch_size': {
                'values': [64, 32, 16, 8]
            },
            'learning_rate': {
                'values': [1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
            },
            'optimizer': {
                'values': ['adam', 'rmsprop', 'sgd']
            },
            'use_feature_extract': {
                'values': [True, False]
            },
            'resnet_type': {
                'values': ['resnet18', 'resnet34', 'resnet50']
            }
        }
    }
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
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=100, epochs=config.epochs,
    #                                          steps_per_epoch=len(dataloaders['train']))
    
    valid_loss_history = []
    valid_map_history = []

    # Run train loop
    start_time = time.time()
    num_steps = len(dataloaders['train'].dataset) // config.batch_size
    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_loss = 0.0
        batch_map = 0.0
        
        for step, (inputs, labels) in enumerate(dataloaders['train']):
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
                #scheduler.step()
            
            batch_loss += loss.item() * inputs.size(0)
            batch_map += calculate_map(outputs, labels)

            if step % num_steps == 0:
                time_taken = time.time() - start_time
                valid_loss = 0.0
                valid_map = 0.0
                    
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item() * inputs.size(0)
                        valid_map += calculate_map(outputs, labels)
                        
                valid_loss /= len(dataloaders['valid'].dataset)
                valid_loss_history.append(valid_loss)                
                valid_map /= len(dataloaders['valid'].dataset)
                valid_map_history.append(valid_map)
                
        total_iteration = len(dataloaders['train'].dataset)
        epoch_loss = batch_loss / total_iteration
        mean_valid_loss = np.mean(valid_loss_history)
        epoch_map = batch_map / total_iteration
        mean_valid_map = np.mean(valid_map_history)
        
        if PRINT_STATUS:
            print_status_bar(epoch,
                             config.epochs,
                             step * config.batch_size,
                             total_iteration,
                             epoch_loss,
                             epoch_map,
                             mean_valid_loss,
                             mean_valid_map,
                             time_taken)
        
def print_status_bar(epoch, total_epoch, iteration, total, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    end = "" if iteration < total else "\n"
    print(f"At epoch: {epoch}/{total_epoch} - iteration: {iteration}/{total}" +
        "- train loss: {train_loss:.4f} - " + f"train map: {train_map:.4f}" +
        "- valid loss: {valid_loss:.4f} - " + f"valid map: {valid_map:.4f}" +
        "- time spent: " + f"{time_taken:.2f}" + "\n" + end)
          
def calculate_map(outputs, labels):
    top_k = torch.topk(outputs, 5)
    preds = top_k.indices.detach().cpu().numpy()
    corrects = labels.detach().cpu().numpy()
    return ml_metrics.mapk([[x] for x in corrects], preds, k=5)
                
if __name__ == "__main__":
    main()