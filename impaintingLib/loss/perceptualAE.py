import impaintingLib as imp
import os 
import torch

modelPath = "./modelSave/perceptualAE.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainAE(model):
    optimizer      = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    trainloader, testloader = imp.data.getFaces(shuffle=False,doNormalize=False)
    criterions     = [(1,torch.nn.L1Loss())]
    
    visu           = imp.utils.Visu(runName = "AutoEncoder", expeName="PerceptualLossModel")
    alterFunc = imp.mask.Alter().addChannel
    
    for i in range(300):
        visuFuncs = [visu.board_loss_train]
        imp.process.train(model, optimizer, trainloader, criterions, epochs=50, alter=alterFunc, visuFuncs=visuFuncs)
        torch.save(model.state_dict(), modelPath)
        
        visuFuncs = [visu.board_plot_img]
        imp.process.test(model, testloader, visuFuncs=visuFuncs)
    
    return model

def getTrainedModel():
    model = imp.model.AutoEncoder().to(device)
    
    if os.path.exists(modelPath) :
        model.load_state_dict(torch.load(modelPath))
        model.eval()
    else :
        model = trainAE(model)
        
    return model
        
def perceptualAE(x, y):
    getTrainedModel()