from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch

class Visu :

    def __init__(self, runName = "default", save = False):
        self.runName = runName
        self.path = "./output/" + runName + ".png"
        self.save = save
        
        self.count    = 0
        self.gridSize = 16
        self.figSize  = (40,30)

    def plot_img(self,images):
        self.count += 1
        images = torch.clip(images[:self.gridSize],0,1)
        img_grid = make_grid(images)
        plt.figure(figsize=self.figSize)
        plt.imshow(img_grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        
        if self.save :
            variable_path = self.path[:-4] + str(self.count) + self.path[-4:] 
            plt.savefig(variable_path)
        else : 
            plt.savefig(self.path)
            
    def plot_all_img(self,**kwargs):
        self.plot_img(kwargs["x"])
        self.plot_img(kwargs["x_prime"])
        self.plot_img(kwargs["x_hat"])
        
    def plot_last_img(self,**kwargs):
        self.plot_img(kwargs["x_hat"])
        
    def board_plot_last(self,**kwargs):
        images_prime = kwargs["x_prime"].cuda()
        images_hat   = kwargs["x_hat"].cuda()
        
        images = torch.cat((images_prime[:self.gridSize],images_hat[:self.gridSize]))
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        
        writer  = SummaryWriter("runs/" + self.runName)
        writer.add_image("Altered / Ouput",img_grid)
        writer.close


# Visualisation
# courbe evolution loss