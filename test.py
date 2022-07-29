import numpy as np
from PIL import Image,ImageDraw

# image = Image.open("pp-chat.png").convert("L")
# image.show()
# np_array = np.array(image)
# print(np_array.shape)

array = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
array = (array + 1) * 25
array = array.astype(np.uint8)
print(array)

pil_image=Image.fromarray(array)
# draw = ImageDraw.Draw(pil_image)
# draw.line((0,0,0,0), fill=128)
array = np.array(pil_image)
print(array)