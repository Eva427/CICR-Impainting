import gradio as gr
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = []
models.append(imp.model.UNet(22, netType="partial")) # , convType="gated"
models[0].load_state_dict(torch.load('./modelSave/partial_22channels_highres.pth', map_location=torch.device('cpu')))

#models.append(imp.model.SubPixelNetwork(3))
#models[0].load_state_dict(torch.load('./modelSave/gated/impainter.pth', map_location=torch.device('cpu')))
#models[1].load_state_dict(torch.load('./modelSave/gated/augmenter.pth', map_location=torch.device('cpu')))

# resize = (64,64)
# resize = (192,192)
resize = (256,256)

def classify(x):
    xNormalized = imp.data.normalize(x)
    modelPath = "./modelSave/classifierUNet.pth"
    classif = imp.model.ClassifierUNet()
    classif.load_state_dict(torch.load(modelPath,map_location=torch.device('cpu')))
    normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
    classifiedImage = classif(normalized)
    return torch.cat((xNormalized,classifiedImage),dim=1)

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
    image = image.view(1,c,w,h)
    #image = imp.data.normalize(image)
    image = classify(image)

    mask  = mask[:1]
    image = imp.mask.propagate(image,mask)
    
    # alter = imp.mask.Alter(min_cut=2, max_cut=25)
    # image = alter.squareMask(image)

    return image

def predict(image):
    for model in models:
        image = model(image)
        
    # image = models[0](image)

    image = imp.data.inv_normalize(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def impaint(input):
    image_prime = convertImage(input)
    image_hat = predict(image_prime)
    image_hat = transforms.ToPILImage()(image_hat[0])

    # image_prime = imp.data.inv_normalize(image_prime)
    # image_prime = torch.clip(image_prime,0,1)
    # image_prime = transforms.ToPILImage()(image_prime[0])
    # image_prime.show()
    # image_hat.show()
    
    return image_hat

css = ".output_image {height: 40rem !important; width: 100% !important;}"
interface = gr.Blocks(css=css)

with interface:
    with gr.Row():
        with gr.Column():
            img = gr.Image(
                tool="sketch", source="upload", label="Input", type="pil"
            )
            with gr.Row():
                btn = gr.Button("Run")
        with gr.Column():
            img2 = gr.Image(interactive=True, label="Output")
            #img3 = gr.Image()

    btn.click(fn=impaint, inputs=img, outputs=img2)
    #btn.click(fn=fn, inputs=img, outputs=[img2, img3])


interface.launch()