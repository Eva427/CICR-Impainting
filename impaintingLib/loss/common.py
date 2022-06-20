import torch 

####################################
# Total Variation loss
####################################

def totalVariation(x_hat,x=0):
    loss = torch.mean(torch.abs(x_hat[:, :, :, :-1] - x_hat[:, :, :, 1:])) + \
            torch.mean(torch.abs(x_hat[:, :, :-1, :] - x_hat[:, :, 1:, :]))
    return torch.mean(loss)