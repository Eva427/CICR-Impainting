import torch 

# Total Variation loss
def totalVariationLoss(x):
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return torch.mean(loss)