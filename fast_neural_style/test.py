import torch
from torchvision import models
from fast_neural_style.neural_style.vgg import Vgg16
from PIL import Image
from fast_neural_style.neural_style.neural_style import train, stylize
import os
from torchvision import transforms
import torch
# model_vgg = Vgg16()
from fast_neural_style.neural_style.MyDataSet import MyDataSet
import fast_neural_style.neural_style.utils as utils

from torch.utils.data import DataLoader
import numpy as np

# im = Image.open("../Data/Driving/RGB_cleanpass/left/0400.png")
im = Image.open("images/content-images/amber.jpg")
# print(type(im))
# im.show()


# image_size = 256
#
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
# ])
#
# dataset_path = "../Data"
# style_image_path = "images/style-images/mosaic.jpg"
# model_dir = "../fast_neural_style/models/"
# videos_list = os.listdir("../Data")
# videos_list = os.listdir(dataset_path)
# train_dataset = {}
# train_loader = {}

# train(dataset_path, style_image_path, model_dir, 1, epochs=200, batch_size=1, log_interval=30)


# for video_name in videos_list:
#     video_dataset_path = os.path.join(dataset_path, video_name)
#     train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
#     train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size= 1)
#
#
# videos_list = os.listdir("../Data")
# counter = 0

# for video_name in videos_list:
#     for batch_id, (frame_left,frame_right) in enumerate(train_loader[video_name]):
#         # frame_left, frame_right = data
#
#         frame_left = frame_left.clone().clamp(0, 255).numpy()
#         frame_left = frame_left.transpose(2, 3, 1, 0).astype("uint8")
#         frame_left = frame_left[:, :, :, 0]
#         frame_left = Image.fromarray(frame_left)
#         counter += 1
#     frame_left.show()
# print(counter)


# model = "models/myModel.pth"
# has_cuda = 1
#
# left_frame_stylized = stylize(has_cuda, im, model)
# stylized_frame = left_frame_stylized.clone().clamp(0, 255).numpy()
# stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
# img = Image.fromarray(stylized_frame)
# img.show()
#

flow_file = open("../Data/Monkaa/optical_flow/forward/0048.pfm")
# print(type(flow_file))
(a,b) = utils.load_pfm(flow_file)
print(a.shape)
print(b)
