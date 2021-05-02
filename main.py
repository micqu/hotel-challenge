import numpy
from skimage import io, transform
from sklearn import preprocessing
from hotel_dataset import HotelImagesDataset
from torch import nn
import torch.optim as optim
import torch
import model
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def encode_labels(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['hotel_id'])
    df['label'] = le.transform(df['hotel_id'])
    return df, le

def main():    
    
    csv_file = 'data/train.csv'
    df = pd.read_csv(csv_file)
    df, le = encode_labels(df)
        
    hotel_dataset = HotelImagesDataset(df,
                                       root_dir='data/train_images/0',
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Resize(32),
                                           #  transforms.RandomCrop(224,),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
    sample = hotel_dataset[0]
    
    criterion = nn.CrossEntropyLoss()
    net = model.Net(880, [256, 256], 7770, 0.5)    
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    trainloader = torch.utils.data.DataLoader(hotel_dataset, batch_size=1,
                                          shuffle=True, num_workers=1)
    
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            topk = torch.topk(outputs, 5)
            hotel_ids = le.inverse_transform(topk.indices.detach().squeeze().numpy())
            print(hotel_ids)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    main()