from impaintingLib import *
from torch.utils.tensorboard import SummaryWriter

### Visualization :
# - plot_all_img 
# - plot_last_img
# - board_plot_last

### Alteration :
# - squareMask

### Models : 
# - AutoEncoder
# - UNet
# - UNetPartialConv
# - SubPixelNetwork

### Loss : 
# - torch.nn.L1Loss()
# - torch.nn.L2Loss()
# - perceptual_loss
# - totalVariationLoss

torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**2)
print(torch.cuda.memory_reserved() / 1024**2)

# -------------- Parameters

trainloader, testloader = getFaces()

runName   = "AutoEncoder Perceptual Epo10"
model     = AutoEncoder().to(device)

optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
criterions =  [ #totalVariationLoss,
                #torch.nn.L1Loss(),
                perceptual_loss ]

# -------------- Process

# Train
alterFunc = Alter(min_cut=4, max_cut=60).squareMask
visuFunc  = None
train(model, optimizer, trainloader, criterions, epochs=10, alter=alterFunc, visu=visuFunc, runName=runName)

# Test
alterFunc = alterFunc
visuFunc  = Visu(runName = runName, save=False).board_plot_last
test(model, testloader, alter=alterFunc, visu=visuFunc)

# -------------- Display

# Tensor Board Model
writer = SummaryWriter("runs/" + runName)
example_input, _ = next(iter(trainloader))
writer.add_graph(model,example_input.cuda())
writer.close()