import gradio as gr
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms
from PIL import Image

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#models.append(imp.model.SubPixelNetwork(3))
#models[0].load_state_dict(torch.load('./modelSave/gated/impainter.pth', map_location=torch.device('cpu')))
#models[1].load_state_dict(torch.load('./modelSave/gated/augmenter.pth', map_location=torch.device('cpu')))

factorResize = 7
resize = (64*factorResize, 64*factorResize)

# Impainter
impainter = imp.model.UNet(22, netType="partial") # , convType="gated"
if factorResize > 1 : 
    impainter_weight_path = './modelSave/partial_22channels_highres.pth'
else : 
    impainter_weight_path = './modelSave/partial_22channels.pth'
impainter.load_state_dict(torch.load(impainter_weight_path, map_location=torch.device('cpu')))

# Classifier
classifier_weight_path = "./modelSave/classifierUNet.pth"
classif = imp.model.ClassifierUNet()
classif.load_state_dict(torch.load(classifier_weight_path,map_location=torch.device('cpu')))

# Enhancer
enhancer_weight_path = 'modelSave/RRDB_ESRGAN_x4.pth'
enhancer = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
enhancer.load_state_dict(torch.load(enhancer_weight_path,map_location=torch.device('cpu')))

def convertImage(input):
    image, mask = input["image"], input["mask"]
    image = transforms.ToTensor()(image)
    mask  = transforms.ToTensor()(mask)

    c,w,h = image.shape
    min_dim = min(w,h)
    image = transforms.CenterCrop((min_dim,min_dim))(image)
    mask = transforms.CenterCrop((min_dim,min_dim))(mask)
    image = transforms.Resize(resize)(image)
    mask  = transforms.Resize(resize)(mask)
    c,w,h = image.shape

    x = image.view(1,c,w,h)
    mask  = mask[:1]
    xNormalized = imp.data.normalize(x)
    x_prime = imp.mask.propagate(xNormalized,mask)

    normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
    classifiedImage = classif(normalized)
    x_prime2 = torch.cat((x_prime,classifiedImage),dim=1)

    colors = imp.loss.generate_label(classifiedImage,w)
    return x_prime2,colors

def predict(image):
    image = impainter(image)
    image = imp.data.inv_normalize(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def impaint(input):
    image_prime,colors = convertImage(input)
    image_hat = predict(image_prime)
    #image_hat = enhance(image_hat)
    image_hat = transforms.ToPILImage()(image_hat[0])
    colors = transforms.ToPILImage()(colors[0])

    # image_prime = imp.data.inv_normalize(image_prime)
    # image_prime = torch.clip(image_prime,0,1)
    # image_prime = transforms.ToPILImage()(image_prime[0])
    # image_prime.show()
    # image_hat.show()
    
    return image_hat,colors

def wtf(input):
    image, mask = input["image"], input["mask"]
    image.show()
    mask.show()

def replace():
    # path = html.split("'")[1]
    # path = path.split("/")[1]
    image = Image.open("current.png")
    return image

css = ".output_image {height: 40rem !important; width: 100% !important;}"
css = ""
interface = gr.Blocks(css=css)

with interface:
    with gr.Row():
        with gr.Column():
            img = gr.Image(
                tool="sketch", source="upload", label="Input", type="pil"
            )
            with gr.Row():
                btn = gr.Button("Run")
                btnChangeImg = gr.Button("Load")
        with gr.Column():
            img3 = gr.Image(label="Classifer", interactive=True, tool="sketch", type="pil")
            html = gr.HTML(visible=False, value="<img src='file/current.png' width='200px'>")
            btn3 = gr.Button("Wtf")
        with gr.Column():
            img2 = gr.Image(label="Output")

    #btn.click(fn=impaint, inputs=img, outputs=img2)
    btn.click(fn=impaint, inputs=img, outputs=[img2, img3])
    btnChangeImg.click(fn=replace, inputs=None, outputs=img3)
    btn3.click(fn=wtf, inputs=img3, outputs=None)

interface.launch()