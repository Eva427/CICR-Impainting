from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

import torch
import os

class Visu :

    def __init__(self, expeName = "default", runName = "default", save = False):
        self.runName  = runName
        self.expeName = expeName
        self.path = "./output/{}/{}.png".format(expeName,runName)
        self.save = save
        
        self.count    = 0
        self.gridSize = 16
        self.figSize  = (80,60)
        
    #--------- PLOT and save

    def plot_img(self,images,**kwargs):
        self.count += 1
        images = torch.clip(images[:self.gridSize],0,1)
        img_grid = make_grid(images)
        plt.figure(figsize=self.figSize)
        plt.imshow(img_grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        
        if self.save :
            dir_path = "/".join(self.path.split("/")[:-1])
            if os.path.exists(dir_path) :
                os.makedirs(dir_path)
                
            variable_path = self.path[:-4] + str(self.count) + self.path[-4:] 
            plt.savefig(variable_path)
            

    def plot_original_img(self,**kwargs):
        self.plot_img(kwargs["x"])
        
    def plot_altered_img(self,**kwargs):
        self.plot_img(kwargs["x_prime"])
        
    def plot_res_img(self,**kwargs):
        self.plot_img(kwargs["x_hat"])
        
    def plot_all_img(self,**kwargs):
        self.plot_original_img(kwargs)
        self.plot_altered_img(kwargs)
        self.plot_res_img(kwargs)
        
    #--------- BOARD
        
    def board_plot_last_img(self,**kwargs):
        images_prime = kwargs["x_prime"].cuda()
        images_hat   = kwargs["x_hat"].cuda()
        
        images = torch.cat((images_prime[:self.gridSize],images_hat[:self.gridSize]))
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        
        writer  = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        
        images = torch.cat((images_prime[:self.gridSize],images_hat[:self.gridSize]))
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        writer.add_image("Altered",img_grid)
        
        
        images = images_hat[:self.gridSize]
        images = torch.clip(images,0,1)
        img_grid = make_grid(images)
        writer.add_image("Output",img_grid)
        
        writer.close
        
    def board_loss_train(self,**kwargs):
        running_loss = kwargs["running_loss"]
        epoch        = kwargs["epoch"]
        
        writer = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        writer.add_scalar("training loss", mean(running_loss), epoch)
        writer.close()
        
    def board_loss_test(self,**kwargs):
        running_loss = kwargs["running_loss"]
        
        writer = SummaryWriter("runs/{}/{}".format(self.expeName,self.runName))
        writer.add_text("testing loss", str(mean(running_loss)))
        writer.close()
        
        
    def none(self,**kwargs):
        pass