from impaintingLib import *

trainloader, testloader = getCeleba()

model     = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
loss      = torch.nn.L1Loss()

visuFunc = Visu(path = "./output/default.png").plot_imgs
alterFunc = Alter(min_cut=15, max_cut=45).squareMask

train(model, optimizer, trainloader, epochs=3, alter=alterFunc, visu=visuFunc)


# images, labels = next(iter(trainloader))
# images = squareMask(images)
# plot_img(images[:8])