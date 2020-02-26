import numpy as np
import os

import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def damaged_good_model(imagenet_weights=True):
    model = models.resnet34(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                              ('drop1', nn.Dropout(0.5)),
                              ('fc1', nn.Linear(512, 2)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return model

def front_rear_side_model(imagenet_weights=True):
    model = models.resnet18(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                            ('drop1', nn.Dropout(0.5)),
                            ('fc1', nn.Linear(512, 3)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return model

def minor_moderate_severe_model(imagenet_weights=True):
    model = models.resnet18(pretrained=imagenet_weights)
    model.fc = nn.Sequential(OrderedDict([
                            ('drop1', nn.Dropout(0.25)),
                            ('fc1', nn.Linear(512, 3)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return model

def transform_function():
    image_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    return image_transform

def preprocess_image(img):
    image_transform = transform_function()
    
    img_for_model = image_transform(img).float()
    img_for_model = Variable(img_for_model,requires_grad=True)
    img_for_model = img_for_model.unsqueeze(0).to(device)
    return img_for_model

def load_models(models_path,damage=True , section=True , intensity=True):
    #print(device)
    damage_model, section_model, intensity_model = None, None, None
    if damage:
        damage_model = damaged_good_model(imagenet_weights=False)
        damage_model.load_state_dict(torch.load(os.path.join(models_path,'damaged_good_model.pth'),map_location=device))
        damage_model.to(device)
        #print('damage_model loaded')
    if section:
        section_model = front_rear_side_model(imagenet_weights=False)
        section_model.load_state_dict(torch.load(os.path.join(models_path,'front_rear_side_model.pth'),map_location=device))
        section_model.to(device)
        #print('side_model loaded')
    if intensity:
        intensity_model = minor_moderate_severe_model(imagenet_weights=False)
        intensity_model.load_state_dict(torch.load(os.path.join(models_path,'minor_moderate_severe_model.pth'),map_location=device))
        intensity_model.to(device)
        #print('intensity_model loaded')
    
    return [damage_model, section_model, intensity_model]

def run_models(img, list_models):
    
    output_dict = {}
    
    damage_classes = {0:'damaged', 1:'good'}
    section_classes = {0:'front', 1:'rear', 2:'side'}
    intensity_classes = {0:'minor', 1:'moderate', 2:'severe'}
    
    damage_model, section_model, intensity_model = list_models
    with torch.no_grad():
        if damage_model!=None:
            damage_model.eval()
            output = torch.softmax(damage_model(img),1)
            score, class_predicted = torch.max(output.data, 1)
            output_dict['damage']=[{'class':damage_classes[class_predicted.item()],
                                     'score':round(score.item(),4)}]            
            if (section_model!=None) & (output_dict['damage'][0]['class']==damage_classes[0]):
                section_model.eval()
                output = torch.softmax(section_model(img),1)
                score, class_predicted = torch.max(output.data, 1)
                output_dict['section']=[{'class':section_classes[class_predicted.item()],
                                         'score':round(score.item(),4)}]  
            if (intensity_model!=None) & (output_dict['damage'][0]['class']==damage_classes[0]):
                intensity_model.eval()
                output = torch.softmax(intensity_model(img),1)
                score, class_predicted = torch.max(output.data, 1)
                output_dict['intensity']=[{'class':intensity_classes[class_predicted.item()],
                                         'score':round(score.item(),4)}]  
    return output_dict