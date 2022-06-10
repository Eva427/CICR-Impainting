import torch
from statistics import mean
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

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

def train(model, optimizer, loader, criterions, epochs=5, alter=None, visu=None):

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
            
            for criterion in criterions :
                loss += criterion(x_hat, x)

            running_loss.append(loss.item()/len(criterions))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}, epoch = {epoch}')
            
        if visu:
            visu(x=x, x_prime=x_prime, x_hat=x_hat, epoch=epoch, running_loss=running_loss)

def test(model, testloader, alter=None, visu=None):
    
    with torch.no_grad():
        x, _ = next(iter(testloader))

        if alter :
            x_prime = alter(x)
        else : 
            x_prime = x

        x_hat = model(x_prime.cuda())
    
    if visu:
        visu(x=x, x_prime=x_prime, x_hat=x_hat)
