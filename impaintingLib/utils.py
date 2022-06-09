from torchvision.utils import make_grid
import matplotlib.pyplot as plt
class Visu :

    def __init__(self, path = "./output/default.png"):
        self.path = path

    def plot_img(self,x):
        img_grid = make_grid(x[:16])
        plt.figure(figsize=(20,15))
        plt.imshow(img_grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(self.path)

    def plot_imgs(self,*xs):
        for x in xs :
            self.plot_img(x)

# Visualisation
# courbe evolution loss