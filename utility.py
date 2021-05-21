import torch
import torchvision
import numpy as np
import numbers
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.functional import pad

class AddPadding(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

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