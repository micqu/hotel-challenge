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
import trainer

IMAGE_SIZE = 32

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
    
    # Load the meta data file
    df = pd.read_csv('data/train.csv',)
    df = df.drop(['timestamp'], axis=1)
    df, _ = utility.encode_labels(df)
    num_classes = len(df['label'].value_counts())
    
    # Build the dataset
    train_loader, valid_loader = dl.get_train_valid_loader(
        df,
        data_dir='data/train_images',
        batch_size=config.batch_size,
        image_size=IMAGE_SIZE,
        augment=True,
        random_seed=0
    )
    
    # Make resnet
    model = utility.initialize_resnet(num_classes, config.resnet_type, config.use_feature_extract)
    model = model.to(device)
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = utility.get_model_params_to_train(model, config.use_feature_extract)
    
    # Define criterion + optimizer
    criterion = nn.CrossEntropyLoss()

    if config.optimizer=='sgd':
        optimizer = optim.SGD(params_to_update, lr=config.learning_rate)
    elif config.optimizer=='rmsprop':
        optimizer = optim.RMSprop(params_to_update, lr=config.learning_rate)
    elif config.optimizer=='adam':
        optimizer = optim.Adam(params_to_update, lr=config.learning_rate)

    # Define scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=10, epochs=config.epochs,
                                              anneal_strategy=config.scheduler,
                                              steps_per_epoch=len(train_loader))
    
    trainer.train_model(device=device,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    scheduler=scheduler,
                    epochs=config.epochs,
                    send_to_wandb=True)
    
if __name__ == "__main__":
    main()