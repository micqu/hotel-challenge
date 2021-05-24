from os import path
import torch
import utility
import wandb
import copy
import time
import datetime
import numpy as np
from scipy import io
from tqdm import tqdm

def train_model(device, model, optimizer, criterion, train_loader, valid_loader, net_type,
                scheduler, epochs, send_to_wandb: bool = False, apply_zca_trans: bool = False):

    best_model_wts = copy.deepcopy(model.state_dict())
    valid_loss_min = np.Inf
    since = time.time()
    
    # Apply ZCA if enabled
    if apply_zca_trans:
        zca_data = io.loadmat('./data/zca_data.mat')
        transformation_matrix = torch.from_numpy(zca_data['zca_matrix']).float()
        transformation_mean = torch.from_numpy(zca_data['zca_mean'][0]).float()
        zca = utility.ZCATransformation(transformation_matrix, transformation_mean)
    
    # Use EarlyStopping
    early_stopping = utility.EarlyStopping(patience=5, verbose=True,
                                           path=f'./checkpoints/checkpoint_{net_type}.pt')
    
    # Run train loop
    print(f"Now training {net_type} ...")    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_map = 0.0
        for _, (inputs, labels) in tqdm(enumerate(train_loader)):  
            if apply_zca_trans:
                inputs = zca(inputs) # apply ZCA transformation
            
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
            
            train_loss += loss.item() * inputs.size(0)
            train_map += utility.calculate_map(outputs, labels)
            
        model.eval()
        valid_loss = 0.0
        valid_map = 0.0
        for _, (inputs, labels) in tqdm(enumerate(valid_loader)):
            with torch.no_grad():
                if apply_zca_trans:
                    inputs = zca(inputs) # apply ZCA transformation
                
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
        
        if scheduler is not None:
            scheduler.step()  # step up scheduler

        # deep copy the model if improved
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if send_to_wandb:     
            wandb.log({"train_loss": train_loss,
                       "train_map": train_map,
                       "epoch": epoch,
                       "valid_loss": valid_loss,
                       "valid_map": valid_map})
        
        time_taken = time.time() - since         
        print_status_bar(epoch, epochs, train_loss, train_map,
                         valid_loss, valid_map, time_taken)
        
        # check if we can stop training
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping activated, stopping ...")
            break
        
    print(f"Completed training {net_type}")
    model.load_state_dict(best_model_wts)
    return model

def print_status_bar(epoch, total_epoch, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    time_taken_min = str(datetime.timedelta(seconds=round(time_taken)))
    print(f"At epoch: {epoch}/{total_epoch}" +
        f" - train loss: {train_loss:.4f}" +
        f" - train map: {train_map:.4f}" +
        f" - valid loss: {valid_loss:.4f}" +
        f" - valid map: {valid_map:.4f}" +
        f" - time spent: {time_taken_min}")