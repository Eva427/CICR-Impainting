import torch 

####################################
# Total Variation loss
####################################

def totalVariation(x_hat,x=0):
    loss = torch.mean(torch.abs(x_hat[:, :, :, :-1] - x_hat[:, :, :, 1:])) + \
            torch.mean(torch.abs(x_hat[:, :, :-1, :] - x_hat[:, :, 1:, :]))
    return torch.mean(loss)

def keypointLoss(x,x_hat):
    keypointX = getKeypoints(x)
    keypointX_hat = getKeypoints(x_hat)
    mse = torch.nn.MSELoss()
    loss = mse(keypointX,keypointX_hat)
    if not loss :
        loss = 0
    return loss