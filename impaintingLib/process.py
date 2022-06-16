import torch
from statistics import mean
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

import impaintingLib as imp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pathFromRunName(runName):
    modelSavePrefix = "./modelSave/"
    runName = runName.replace(" ","_")
    path = modelSavePrefix + runName + ".pth"
    return path

def model_save(model, runName):
    path = pathFromRunName(runName)
    torch.save(model.state_dict(), path)

def model_load(model, runName):
    path = pathFromRunName(runName)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def train(model, optimizer, loader, criterions, epochs=5, alter=None, visuFuncs=None):

    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)

        for x, _ in t:
            x = x.to(device)

            if alter :
                x_prime = alter(x)
            else : 
                x_prime = x

            x_hat = model(x_prime.cuda())
            loss  = 0
            
            for coef,criterion in criterions :
                loss += criterion(x_hat, x)*coef

            running_loss.append(loss.item()/len(criterions))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}, epoch = {epoch}/{epochs}')
            
        x       = imp.data.inv_normalize(x)
        x_prime = imp.data.inv_normalize(x_prime)
        x_hat   = imp.data.inv_normalize(x_hat)
            
        if visuFuncs:
            for visuFunc in visuFuncs : 
                visuFunc(x=x, x_prime=x_prime, x_hat=x_hat, epoch=epoch, running_loss=running_loss)

def test(model, loader, alter=None, visuFuncs=None):
    
    with torch.no_grad():
        
        running_loss = []
        t = tqdm(loader)
        for x, _ in t:
            x = x.to(device)

            if alter :
                x_prime = alter(x)
            else : 
                x_prime = x

            x_hat = model(x_prime.cuda())
            loss = imp.loss.perceptual_loss(x,x_hat)
            running_loss.append(loss.item())
            t.set_description(f'testing loss: {mean(running_loss)}')
            
        x       = imp.data.inv_normalize(x)
        x_prime = imp.data.inv_normalize(x_prime)
        x_hat   = imp.data.inv_normalize(x_hat)
    
        if visuFuncs:
            for visuFunc in visuFuncs : 
                visuFunc(x=x, x_prime=x_prime, x_hat=x_hat, epoch=0, running_loss=running_loss)