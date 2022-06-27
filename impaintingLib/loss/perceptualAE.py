import impaintingLib as imp
import os 
import torch

modelPath = "./modelSave/perceptualAE_L1.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gram_matrix(input):
    a, b, c, d = input.size()   # a=batch size(=1)
                                # b=number of feature maps
                                # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d) 
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)

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
    #loss += mse(gram_matrix(x_feats), gram_matrix(y_feats))
    loss = mse(x_feats,y_feats)
        
    return loss
    