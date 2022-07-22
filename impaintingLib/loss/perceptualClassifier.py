import impaintingLib as imp
import os 
import torch
from torchvision import transforms

modelPath = "./modelSave/classifierUNet.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transformer():
    options = []
    options.append(transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225]))
    options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    return transform

def getTrainedModel():
    model = imp.model.ClassifierUNet().to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model
        
def perceptualClassifier(x, y):
    model = getTrainedModel()
    mse = torch.nn.MSELoss()
    
    x = transformer()(x)
    y = transformer()(y)
    
    x_feats = model.getFeatures(x)
    y_feats = model.getFeatures(y)
    
    loss = 0
    loss = mse(x_feats,y_feats)
        
    return loss
    