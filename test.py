import gradio as gr
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = []
models.append(imp.model.UNet(3, netType="partial"))
models.append(imp.model.SubPixelNetwork(3))
models[0].load_state_dict(torch.load('./modelSave/UNet(partial conv2d).pth', map_location=torch.device('cpu')))
models[1].load_state_dict(torch.load('./modelSave/PixelShuffle1.pth', map_location=torch.device('cpu')))


def convertImage(input):
    image, mask = input["image"], input["mask"]
    image = transforms.ToTensor()(image)
    mask  = transforms.ToTensor()(mask)

    c,w,h = image.shape
    mask = mask[:1]
    image = image.view(1,c,w,h)

    image = imp.mask.propagate(image,mask)
    return image

def impaint(input):
    image = convertImage(input)
    for model in models:
        image = model(image)

    image = transforms.ToPILImage()(image[0])
    return image

demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Column():
            img = gr.Image(
                tool="sketch", source="upload", label="Mask", type="pil"
            )
            with gr.Row():
                btn = gr.Button("Run")
        with gr.Column():
            img2 = gr.Image()
            #img3 = gr.Image()

    btn.click(fn=impaint, inputs=img, outputs=img2)
    #btn.click(fn=fn, inputs=img, outputs=[img2, img3])


demo.launch()