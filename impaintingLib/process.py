import torch
from statistics import mean
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import transforms

import numpy as np
import impaintingLib as imp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classif = imp.loss.getTrainedModel()

def pathFromRunName(runName):
    modelSavePrefix = "./modelSave/"
    runName = runName.replace(" ","_")
    path = modelSavePrefix + runName # + ".pth"
    return path

def model_save(models, runName):
    
    # On ne crée et utilise un dossier que si il y'a plusieurs models
    if len(models) < 2:
        dir_path = "./modelSave"
    else : 
        dir_path = pathFromRunName(runName)
        if not os.path.exists(dir_path) :
            os.makedirs(dir_path)
    
    for model in models :  
        path = dir_path + "/" + str(model) + ".pth"
        torch.save(model.state_dict(), path)

def model_load(models, runName):
    
    if len(models) < 2:
        dir_path = "./modelSave"
    else : 
        dir_path = pathFromRunName(runName)
    
    for i,model in enumerate(models) :  
        path = dir_path + "/" + str(model) + ".pth"
        model.load_state_dict(torch.load(path))
        model.eval()
        models[i] = model
    
    return models

def npToTensor(x):
    c,w,h = x.shape
    x = x / c
    x = torch.from_numpy(x).to(device)
    x = torch.reshape(x, (c,1,w,h))
    return x

def simplifyChannels(x):
    x = np.where(x == 3, 0, x) 
    x = np.where(x == 4, 3, x) 
    x = np.where(x == 5, 3, x) 
    x = np.where(x == 6, 4, x) 
    x = np.where(x == 7, 4, x) 
    x = np.where(x == 8, 5, x) 
    x = np.where(x == 9, 5, x) 
    x = np.where(x == 10 , 6, x) 
    x = np.where(x == 11, 7, x) 
    x = np.where(x == 12, 7, x)  
    x = np.where(x > 12, 0, x) 
    return x

def train(models, optimizer, loader, criterions, epochs=5, alter=None, visuFuncs=None, classify=False):
    
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)

        for x, _ in t:
            # x = imp.data.randomTransfo(x)
            x = x.to(device)
            
            if alter :
                x_prime = alter(imp.data.normalize(x)) 
            else : 
                x_prime = imp.data.normalize(x)
            
            if classify :
                normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
                classifiedImage = classif(normalized)
                _,_,w,_ = x.shape
                classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
                classifPlain = simplifyChannels(classifPlain)
                classifPlain = npToTensor(classifPlain)
                classifPlain = classifPlain.float()
                x_prime2 = torch.cat((x_prime,classifPlain),dim=1)
            else :
                x_prime2 = x_prime

            #x = imp.data.crop(x)
            x_hat = x_prime2.cuda()
            for model in models:
                x_hat = model(x_hat)
                
            loss  = 0
            for coef,criterion in criterions :
                loss += criterion(x_hat, x)*coef
                
            #if (x[0] < 0).any() or ((x[0] > 1).any()) :
            #    print("input pas entre 0 et 1")
            #if (x_hat[0] < 0).any() or ((x_hat[0] > 1).any()) :
            #    print("output pas entre 0 et 1")

            running_loss.append(loss.item()/len(criterions))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}, epoch = {epoch}/{epochs}')
        
        # x       = imp.data.inv_normalize(x)
        x_prime = imp.data.inv_normalize(x_prime)
        x_hat   = imp.data.inv_normalize(x_hat)
            
        if visuFuncs:
            for visuFunc in visuFuncs : 
                visuFunc(x=x, x_prime=x_prime, x_hat=x_hat, epoch=epoch, running_loss=running_loss)

def test(models, loader, alter=None, visuFuncs=None, classify=False):
    
    with torch.no_grad():
        
        running_loss = []
        x, _ = next(iter(loader))
        x = x.to(device)

        if alter :
            x_prime = alter(imp.data.normalize(x)) 
        else : 
            x_prime = imp.data.normalize(x)

        if classify :
            normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
            classifiedImage = classif(normalized)
            _,_,w,_ = x.shape
            classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
            classifPlain = simplifyChannels(classifPlain)
            classifPlain = npToTensor(classifPlain)
            classifPlain = classifPlain.float()
            x_prime2 = torch.cat((x_prime,classifPlain),dim=1)
        else :
            x_prime2 = x_prime

        x_hat = x_prime2.cuda()
        for model in models:
            x_hat = model(x_hat)
        
        # x       = imp.data.inv_normalize(x)
        x_prime = imp.data.inv_normalize(x_prime)
        x_hat   = imp.data.inv_normalize(x_hat)
    
        if visuFuncs:
            for visuFunc in visuFuncs : 
                visuFunc(x=x, x_prime=x_prime, x_hat=x_hat, epoch=0, running_loss=running_loss)