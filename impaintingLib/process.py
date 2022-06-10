import torch
from statistics import mean
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, loader, criterion, epochs=5, alter=None, visu=None, runName=""):

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
            loss = criterion(x_hat, x)

            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}, epoch = {epoch}')
            
        if runName:
            writer = SummaryWriter("runs/" + runName)
            writer.add_scalar("training loss", mean(running_loss), epoch)
            writer.close()

        if visu:
            visu(x=x, x_prime=x_prime, x_hat=x_hat, epoch=epoch)

def pre_train(model, path):
    return model.load_state_dict(torch.load(path))

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
