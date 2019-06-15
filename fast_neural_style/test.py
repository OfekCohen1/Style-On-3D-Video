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
import fast_neural_style.neural_style.utils_dataset as utils_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm



dataset_path = "../Data/Monkaa"
dataset_path_train = "../../Data/Monkaa"
style_image_path = "images/style-images/mosaic.jpg"
model_dir = "../fast_neural_style/models/"
checkpoint_model_dir = "../fast_neural_style/models/checkpoint_models/"
has_cuda = 1
# videos_list = os.listdir("../Data")
# videos_list = os.listdir(dataset_path)
# train_dataset = {}
# train_loader = {}

image_size = 256

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])
train_dataset = MyDataSet(dataset_path, transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir, log_interval=5, checkpoint_interval=4000,
      content_weight=1, style_weight=5)

# for video_name in videos_list:
#     video_dataset_path = os.path.join(dataset_path, video_name)
#     train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
#     train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size=1)



# counter = 0
# for frames in tqdm(train_loader):
#     (frames_curr, frames_next) = frames
#     frame_curr_left = frames_curr[0]
#     frame_curr_left = 255 * frame_curr_left.clone().clamp(0, 255).numpy()
#     frame_left = frame_curr_left.transpose(2, 3, 1, 0).astype("uint8")
#     frame_left = frame_left[:, :, :, 0]
#     frame_left = Image.fromarray(frame_left)
#     counter += 1
#     if counter == 124:
#         break
# frame_left.show()
# print(counter)


# model = "models/myModel.pth"
# has_cuda = 1
# im = Image.open("images/content-images/amber.jpg")
# left_frame_stylized = stylize(has_cuda, im, model)
# stylized_frame = left_frame_stylized.clone().clamp(0, 255).numpy()
# stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
# img = Image.fromarray(stylized_frame)
# img.show()
#


# im2 = Image.open("images/content-images/0002.webp").convert("RGB")


# print('hello')
# pic_direcs_list = utils_dataset.get_pics_direcs("../Data/Monkaa")
# num_frame = 0
# frame_path_curr_left = pic_direcs_list[num_frame][0]
# frame_path_curr_right = pic_direcs_list[num_frame][1]
#
# frame_name_num = os.path.basename(frame_path_curr_left)  # ie "0001.png"
# suffix = os.path.splitext(frame_path_curr_left)[1]  # .png
# direc_without_num_frame_left = frame_path_curr_left.replace(frame_name_num, '')
# direc_without_num_frame_right = frame_path_curr_right.replace(frame_name_num, '')
# frame_name_string = frame_name_num.replace(suffix, '')  # 0001
#
# num_digits = len(frame_name_string)  # 0001 -> num_digits = 4
# int_curr = int(frame_name_string)
# int_next = int_curr + 1  # n+1 as int
# string_next = str(int_next)
# num_digits_next = len(string_next)
# string_next = '0' * (num_digits - num_digits_next) + string_next + suffix  # 0002.png
#
# frame_path_next_left = os.path.join(direc_without_num_frame_left, string_next)
# frame_path_next_right = os.path.join(direc_without_num_frame_right, string_next)
# frame_next_left = Image.open(frame_path_next_left)
# frame_next_right = Image.open(frame_path_next_right)
# print(type(frame_next_left))

###### Optical flow Test ########
# img = Image.open("../Data/Monkaa/RGB_cleanpass/left/0049.png")
# flow_path = "../Data/Monkaa/optical_flow_resized/forward/0049.flo"
# height = 256
# width = 256
# img = img.resize((width, height), Image.ANTIALIAS)
# new_image, mask = utils.apply_flow(img, flow_path)
# # new_image, mask = show_optical_flow()
# Image.fromarray(new_image).show()
# Image.fromarray(mask*255).show()
