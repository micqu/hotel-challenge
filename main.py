import numpy as np
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from hotel_dataset import HotelImagesDataset
from torch import nn
import torch.optim as optim
import torch
import torchvision
import model
import os
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
import copy
import ml_metrics

IMAGE_SIZE = 224
BATCH_SIZE = 1
NUM_EPOCHS = 15
FEATURE_EXTRACT = True

def encode_labels(df):
    le = LabelEncoder()
    le.fit(df['hotel_id'])
    df['label'] = le.transform(df['hotel_id'])
    df = df.drop(['hotel_id'], axis=1)
    return df, le

def main():    
    df = pd.read_csv('data/train.csv')
    df = df.groupby('hotel_id').filter(lambda x : len(x)>1) #remove hotel ids with only 1 sample
    df, label_encoder = encode_labels(df)
    
    y = df.label
    X = df.drop(['label', 'timestamp'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, stratify=y)
    
    df_train = X_train.merge(y_train, left_index=True, right_index=True)
    df_valid = X_valid.merge(y_valid, left_index=True, right_index=True)
    
    df_train = df_train.iloc[:1000,:] # only first 1000 rows
    df_valid = df_valid.iloc[:1000,:]
    
    data_dir = 'data/train_images'
    data_files = {'train': df_train, 'valid': df_valid}
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
    
    image_datasets = {x: HotelImagesDataset(data_files[x],
                                            root_dir=data_dir,
                                            transform=data_transforms[x])
                      for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
        
    num_classes = len(df['label'].value_counts())
    model_ft, input_size = initialize_resnet(num_classes, FEATURE_EXTRACT, use_pretrained=True)
    
    # Send model to GPU
    model_ft = model_ft.to(device)
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = model_ft.parameters()
    if FEATURE_EXTRACT:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
    #                                               steps_per_epoch=len(dataloaders['train']),
    #                                                epochs=10)
    model_ft, hist_loss, hist_map = train_model(model_ft, dataloaders, criterion, optimizer,
                                                None, device, label_encoder, num_epochs=NUM_EPOCHS)
    
    ohist_loss = [h.cpu().numpy() for h in hist_loss]
    ohist_map = [h.cpu().numpy() for h in hist_map]
    
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, NUM_EPOCHS+1), ohist_loss, label="loss")
    plt.plot(range(1, NUM_EPOCHS+1), ohist_map, label="map")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, NUM_EPOCHS+1, 1.0))
    plt.legend()
    plt.show()
    
def initialize_resnet(num_classes, feature_extract, use_pretrained=True):
    model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = IMAGE_SIZE
    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, device, label_encoder, num_epochs=10):    
    valid_loss_history = []
    valid_map_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_valid_map = 0.0

    start_time = time.time()
    num_steps = len(dataloaders['train'].dataset) // BATCH_SIZE
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        model.train()
        train_loss = 0.0
        train_map = 0.0
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
            
            train_loss += loss.item() * inputs.size(0)
            train_map += calculate_map(outputs, labels)
            
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
                
                if valid_map > best_valid_map:
                    best_valid_map = valid_map
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                valid_loss_history.append(valid_loss)
                valid_map_history.append(valid_map)

        total_iteration = len(dataloaders['train'].dataset)
        train_loss /= total_iteration
        train_map /= total_iteration
        mean_valid_loss = np.mean(valid_loss_history)
        mean_valid_map = np.mean(valid_map_history)
        print_status_bar(step * BATCH_SIZE,
                         total_iteration,
                         train_loss,
                         train_map,
                         mean_valid_loss,
                         mean_valid_map,
                         time_taken)
        print('Train loss: {:.4f}, map: {:.4f}'.format(train_loss, train_map))
        
    model.load_state_dict(best_model_wts)
    return model, valid_loss_history, valid_map_history

def print_status_bar(iteration, total, train_loss, train_map,
                     mean_valid_loss, mean_valid_map, time_taken):
    pass
    
    
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