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
# resize = (128,128)
resize = (256,256)

def classify(x,w):
    xNormalized = imp.data.normalize(x)
    modelPath = "./modelSave/classifierUNet.pth"
    classif = imp.model.ClassifierUNet()
    classif.load_state_dict(torch.load(modelPath,map_location=torch.device('cpu')))
    normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
    classifiedImage = classif(normalized)
    colors = imp.loss.generate_label(classifiedImage,w)
    return torch.cat((xNormalized,classifiedImage),dim=1),colors

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
    image,colors = classify(image,w)

    mask  = mask[:1]
    image = imp.mask.propagate(image,mask)
    
    # alter = imp.mask.Alter(min_cut=2, max_cut=25)
    # image = alter.squareMask(image)

    return image,colors

def predict(image):
    for model in models:
        image = model(image)
        
    # image = models[0](image)

    image = imp.data.inv_normalize(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def enhance(image):
    image = transforms.Resize((64,64))(image)
    model_path = 'modelSave/RRDB_ESRGAN_x4.pth'
    model = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    return model(image)

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
            img3 = gr.Image(interactive=True, label="Classifer")
            html = gr.HTML(visible=False, value="<img src='file/linkdin.jpg' width='200px'>")
        with gr.Column():
            img2 = gr.Image(interactive=True, label="Output")

    #btn.click(fn=impaint, inputs=img, outputs=img2)
    btn.click(fn=impaint, inputs=img, outputs=[img2, img3])


interface.launch()