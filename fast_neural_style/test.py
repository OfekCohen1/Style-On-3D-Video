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
from torch.utils.data import DataLoader
import numpy as np

im = Image.open("images/content-images/amber.jpg")
# print(type(im))
# im.show()


image_size = 256

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
#
dataset_path = "../Data"

videos_list = os.listdir(dataset_path)
train_dataset = {}
train_loader = {}
for video_name in videos_list:
    video_dataset_path = os.path.join(dataset_path, video_name)
    train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
    train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size= 1)

counter = 0
for video_name in videos_list:
    for batch_id, (frame_left,frame_right) in enumerate(train_loader[video_name]):
        # frame_left, frame_right = data

        frame_left = frame_left.clone().clamp(0, 255).numpy()
        frame_left = frame_left.transpose(2, 3, 1, 0).astype("uint8")
        frame_left = frame_left[:, :, :, 0]
        frame_left = Image.fromarray(frame_left)
        counter += 1
    # frame_left.show()
print(counter)
# train(dataset_path, style_image_path, model_dir, 1, batch_size=1, log_interval=1)

# model = "models/myModel.pth"
# has_cuda = 1
#
# left_frame_stylized = stylize(has_cuda, im, model)
# stylized_frame = left_frame_stylized.clone().clamp(0, 255).numpy()
# stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
# img = Image.fromarray(stylized_frame)
# img.show()

# style_image_path = "../fast_neural_style/images/style-images/mosaic.jpg"
# model_dir = "../fast_neural_style/models/"
# videos_list = os.listdir("../Data")
# print(type(videos_list))