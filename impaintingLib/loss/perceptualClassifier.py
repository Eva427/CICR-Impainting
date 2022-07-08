import impaintingLib as imp
import os 
import torch

modelPath = "./modelSave/classifierUNet.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
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
    x_feats = model(x)
    y_feats = model(y)
    # generate_label(labels_predict,size)[0]
    
    loss = 0
    loss = mse(x_feats,y_feats)
        
    return loss
    