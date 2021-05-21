import torch
import torchvision
from sklearn.preprocessing import LabelEncoder

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def encode_labels(df):
    le = LabelEncoder()
    le.fit(df['hotel_id'])
    df['label'] = le.transform(df['hotel_id'])
    df = df.drop(['hotel_id'], axis=1)
    return df, le

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
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False