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

PRINT_STATUS = False # set to False to avoid print messages
USE_AMP = False # set to True to use NVIDIA apex 16-bit precision

def main():
    pass
        
def print_status_bar(epoch, total_epoch, train_loss, train_map,
                     valid_loss, valid_map, time_taken):
    print(f"At epoch: {epoch}/{total_epoch}" +
        f"- train loss: {train_loss:.4f} - " + f"train map: {train_map:.4f}" +
        f"- valid loss: {valid_loss:.4f} - " + f"valid map: {valid_map:.4f}" +
        "- time spent: " + f"{time_taken:.2f}" + "\n")
                
if __name__ == "__main__":
    main()