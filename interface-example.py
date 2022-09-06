import gradio as gr
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

html_file = open("test2.html", 'r', encoding='utf-8')
html = html_file.read() 

css = ".output-image, .input-image {height: 40rem !important; width: 100% !important;}"
#css = "@media screen and (max-width: 600px) { .output_image, .input_image {height:20rem !important; width: 100% !important;} }"
# css = ".output_image, .input_image {height: 600px !important}"
# css = ".image-preview {height: auto !important;}"
interface = gr.Blocks(css=css)

with interface:
    with gr.Row():
        with gr.Column():
            img = gr.Image(tool="sketch", source="upload", label="Input", type="pil")
            with gr.Row():
                btnSegment = gr.Button("Segment")
        with gr.Column():
            html = gr.HTML(value=html)
        with gr.Column():
            imgOutput = gr.Image(label="Output",interactive=False)
            with gr.Row():
                btnRun = gr.Button("Run")

interface.launch()