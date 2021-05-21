import numpy as np
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from hotel_dataset import HotelImagesDataset
from torch import nn
import torch.optim as optim
import torch
import torchvision
import os
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
import copy
import ml_metrics
import wandb

IMAGE_SIZE = 224

def encode_labels(df):
    le = LabelEncoder()
    le.fit(df['hotel_id'])
    df['label'] = le.transform(df['hotel_id'])
    df = df.drop(['hotel_id'], axis=1)
    return df, le

def build_dataset(batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    df = pd.read_csv('data/train.csv')
    df = df.groupby('hotel_id').filter(lambda x : len(x)>1) #remove hotel ids with only 1 sample
    df, label_encoder = encode_labels(df)
    num_classes = len(df['label'].value_counts())
        
    y = df.label
    X = df.drop(['label', 'timestamp'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, stratify=y)
    
    df_train = X_train.merge(y_train, left_index=True, right_index=True)
    df_valid = X_valid.merge(y_valid, left_index=True, right_index=True)
    
    df_train = df_train.iloc[:1000,:] # only first 1000 rows
    df_valid = df_valid.iloc[:1000,:]
    
    data_dir = 'data/train_images'
    data_files = {'train': df_train, 'valid': df_valid}
    image_datasets = {x: HotelImagesDataset(data_files[x],
                                            root_dir=data_dir,
                                            transform=data_transforms[x])
                      for x in ['train', 'valid']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}
    
    return dataloaders, label_encoder, num_classes

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
                'values': [20, 50, 100]
            },
            'batch_size': {
                'values': [512, 256, 128, 64, 32, 16]
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
    
def initialize_resnet(num_classes, resnet_type,
                      feature_extract, use_pretrained=True):
    if resnet_type=='resnet18':
        model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
    elif resnet_type=='resnet34':
        model_ft = torchvision.models.resnet34(pretrained=use_pretrained)
    elif resnet_type=='resnet50':
        model_ft = torchvision.models.resnet50(pretrained=use_pretrained)     
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

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
    dataloaders, label_encoder, num_classes = build_dataset(config.batch_size)
    
    # Make resnet
    model = initialize_resnet(num_classes,
                              config.resnet_type,
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
    
    # Define scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=100, epochs=config.epochs,
                                              steps_per_epoch=len(dataloaders['train']))
    
    valid_loss_history = []
    valid_map_history = []

    # Run train loop
    start_time = time.time()
    num_steps = len(dataloaders['train'].dataset) // config.batch_size
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
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
                scheduler.step()
            
            batch_loss += loss.item() * inputs.size(0)
            batch_map += calculate_map(outputs, labels)
            wandb.log({"batch loss":batch_loss})
            wandb.log({"batch map":batch_map})
            
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
                valid_map /= len(dataloaders['valid'].dataset)
                wandb.log({"valid loss":valid_loss})
                wandb.log({"valid map":valid_map})
                
                valid_loss_history.append(valid_loss)
                valid_map_history.append(valid_map)

        total_iteration = len(dataloaders['train'].dataset)
        epoch_loss = batch_loss / total_iteration
        epoch_map = batch_map / total_iteration
        mean_valid_loss = np.mean(valid_loss_history)
        mean_valid_map = np.mean(valid_map_history)
        
        wandb.log({"epoch loss":epoch_loss})
        wandb.log({"epoch map":epoch_map})
        print_status_bar(step * config.batch_size,
                         total_iteration,
                         epoch_loss,
                         epoch_map,
                         mean_valid_loss,
                         mean_valid_map,
                         time_taken)
        
def print_status_bar(iteration, total, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    end = "" if iteration < total else "\n"
    print(f"{iteration}/{total} - train loss: {train_loss:.4f} - " + 
        f"train map: {train_map:.4f} - valid loss: {valid_loss:.4f} - " +
        f"valid map: {valid_map:.4f} - " + f"time spent: " +
        f"{time_taken:.2f}" + "\n" + end)
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
      
def calculate_map(outputs, labels):
    top_k = torch.topk(outputs, 5)
    preds = top_k.indices.detach().cpu().numpy()
    corrects = labels.detach().cpu().numpy()
    return ml_metrics.mapk([[x] for x in corrects], preds, k=5)
                
if __name__ == "__main__":
    main()