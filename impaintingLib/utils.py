from torchvision.utils import make_grid
import matplotlib.pyplot as plt
class Visu :

    def __init__(self, path = "./output/default.png", save_each = False):
        self.path = path
        self.save_each = save_each
        self.count = 0

    def plot_img(self,x):
        self.count += 1
        img_grid = make_grid(x[:8])
        plt.figure(figsize=(20,15))
        plt.imshow(img_grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        
        if self.save_each :
            variable_path = self.path[:-4] + str(self.count) + self.path[-4:] 
            plt.savefig(variable_path)
        else : 
            plt.savefig(self.path)
            
    def plot_all(self,x,x_prime,x_hat):
        self.plot_img(x[:8])
        self.plot_img(x_prime[:8])
        self.plot_img(x_hat[:8])
        
    def plot_last(self,x,x_prime,x_hat):
        self.plot_img(x_hat[:8])

# Visualisation
# courbe evolution loss