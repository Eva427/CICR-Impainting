import torch
from statistics import mean
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, loader, criterion, epochs=5, alterFunc=None, visu=None, **kwargs):

    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)

        for x, _ in t:
            x = x.to(device)

            if alterFunc :
                x_prime = alterFunc(x)
            else : 
                x_prime = x

            x_hat = model(x_prime)
            loss = criterion(x_hat, x)

            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}, epoch = {epoch}')

        if visu:
            visu(x, x_prime, x_hat)

def pre_train(model, path):
    return model.load_state_dict(torch.load(path))

def test(model, testloader, alterFunc=None, visu=None, **kwargs):
    
    with torch.no_grad():
        x, _ = next(iter(testloader))

        if alterFunc :
            x_prime = alterFunc(x)
        else : 
            x_prime = x

        x_hat = model(x_prime.cuda())
    
    if visu:
        visu(x, x_prime, x_hat)
