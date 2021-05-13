import numpy
from skimage import io, transform
from sklearn import preprocessing
from hotel_dataset import HotelImagesDataset
from torch import nn
import torch.optim as optim
import torch
import torchvision
import model
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def encode_labels(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['hotel_id'])
    df['label'] = le.transform(df['hotel_id'])
    return df, le

def main():    
    csv_file = 'data/train.csv'
    df = pd.read_csv(csv_file)
    df, le = encode_labels(df)
    
    train_size = int(0.8 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_hotel_ds = HotelImagesDataset(train_df,
                                        root_dir='data/train_images/',
                                        transform=transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
    
    val_hotel_ds = HotelImagesDataset(val_df,
                                      root_dir='data/train_images/',
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))

    #net = model.Net(input_dim=44944,
    #                hidden_dims=[256, 256],
    #                output_dim=7770,
    #                dropout_p=0.5)
    #net.to(device)
    net = torchvision.models.resnet34(pretrained=True)
    num_cl = len(df['hotel_id'].value_counts())
    net.fc = nn.Linear(512, num_cl, bias=True)
    net = net.to(device)
    
    train_loader = torch.utils.data.DataLoader(train_hotel_ds, batch_size=64,
                                              shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(val_hotel_ds)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                    steps_per_epoch=len(train_loader), epochs=10)
    
    net.train()
    num_epochs = 1
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        running_loss = 0.0
        i = 0
        for data, target in train_loader:
            # get the data and targets
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # print statistics
            running_loss += loss.detach().item()
            if i % 50 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            i = i+1
    
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = net(data)
            criterion = nn.NLLLoss()
            loss = criterion(torch.log(pred), target)  
            test_loss += loss * len(data)
            
            top_k = torch.topk(pred, 5)
            pred_hotel_ids = le.inverse_transform(top_k.indices.detach().squeeze().cpu().numpy())
            ground_truth = target.cpu().numpy()[0]
            correct += calculate_correctness(ground_truth, pred_hotel_ids)
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def calculate_correctness(truth, pred):
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