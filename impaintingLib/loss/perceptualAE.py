import impaintingLib as imp
import os 
import torch

modelPath = "./modelSave/perceptualAE_L1.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainAE(model):
    optimizer               = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    trainloader, testloader = imp.data.getFaces(shuffle=False,doNormalize=False)
    criterions              = [(1,torch.nn.L1Loss())]
    
    visu = imp.utils.Visu(runName = "TrainPercepLoss", expeName="AEModel")
    
    for i in range(50):
        visuFuncs = []
        imp.process.train(model, optimizer, trainloader, criterions, epochs=10, visuFuncs=visuFuncs)
        torch.save(model.state_dict(), modelPath)
        
        visuFuncs = [visu.board_plot_img]
        imp.process.test(model, testloader, visuFuncs=visuFuncs)
    
    return model

def getTrainedModel():
    model = imp.model.AutoEncoder(3).to(device)
    
    if os.path.exists(modelPath) :
        model.load_state_dict(torch.load(modelPath))
        model.eval()
    else :
        print("ERREUR - Pas de model pré-entrainé disponible")
        model = trainAE(model)
        
    return model
        
def perceptualAE(x, y):
    model = getTrainedModel()
    mse = torch.nn.MSELoss()
    x_feats = model.encoder(x)
    y_feats = model.encoder(y)
    
    loss = 0
    loss = mse(x_feats,y_feats)
        
    return loss
    