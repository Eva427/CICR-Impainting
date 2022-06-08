from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def plot_img(x, path="./output/default.png"):
    img_grid = make_grid(x[:16])
    plt.figure(figsize=(20,15))
    plt.imshow(img_grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(path)

# Visualisation
# courbe evolution loss