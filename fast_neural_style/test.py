import torch
from torchvision import models
from fast_neural_style.neural_style.vgg import Vgg16
from PIL import Image

# model_vgg = Vgg16()

im = Image.open("images/content-images/amber.jpg")
print(type(im))
Image._show(im)
