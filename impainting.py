from impaintingLib import *

trainloader, testloader = getCeleba()

images, labels = next(iter(trainloader))
images = squareMask(images)
plot_img(images[:8])