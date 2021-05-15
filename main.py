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

IMAGE_SIZE = 224
BATCH_SIZE = 4
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
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
        
    num_classes = len(df['label'].value_counts())
    model_ft, input_size = initialize_resnet(num_classes, FEATURE_EXTRACT, use_pretrained=True)
    
    # Send model to GPU
    model_ft = model_ft.to(device)
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if FEATURE_EXTRACT:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                    steps_per_epoch=len(dataloaders['train']),
                                                    epochs=10)
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer,
                                 scheduler, device, label_encoder, num_epochs=NUM_EPOCHS)
    
def initialize_resnet(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    
    model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = IMAGE_SIZE
    
    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, device, label_encoder, num_epochs=10):
    since = time.time()
    valid_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode    
            
            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                top_k = torch.topk(outputs, 5)
                preds = top_k.indices.detach().squeeze().cpu().numpy()
                for i, pred in enumerate(preds):
                    pred_hotel_ids = label_encoder.inverse_transform(pred)
                    truth = labels.cpu().numpy()[i]
                    running_corrects += calculate_correctness(truth, pred_hotel_ids)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_acc_history
      
def set_parameter_requires_grad(model, feature_extracting):
    # Helper function to set requires grad to False
    # if we are feature extracting
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
      
def calculate_correctness(truth, pred):
    # Helper function to calculate corretness
    if pred[0] == truth:
        return 1.0
    elif pred[1] == truth:
        return 0.8
    elif pred[2] == truth:
        return 0.6
    elif pred[3] == truth:
        return 0.4
    elif pred[4] == truth:
        return 0.2
    return 0

if __name__ == "__main__":
    main()