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
import flow_resize_script
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import fast_neural_style.neural_style.losses as losses

# videos_list = os.listdir("../Data")
# videos_list = os.listdir(dataset_path)
# train_dataset = {}
# train_loader = {}
# for video_name in videos_list:
#     video_dataset_path = os.path.join(dataset_path, video_name)
#     train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
#     train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size=1)

image_size = (360, 640)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])
def train_models():
    image_size = (360, 640)
    dataset_path = "../Data/Monkaa"
    style_image_path = "images/style-images/mosaic.jpg"
    model_dir = "../fast_neural_style/models/"
    checkpoint_model_dir = "../fast_neural_style/models/checkpoint_models"
    has_cuda = 1
    #

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=3, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=4000,
          model_filename="model_test_temp_7.5e4_content_1e4_style_8e8_disp_5e3_both_eyes",
          temporal_weight=7.5e4, content_weight=1e4, style_weight=8e8, disp_weight=5e3)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=3, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=4000,
          model_filename="model_test_temp_8e4_content_2e4_style_8e8_disp_5e3_both_eyes",
          temporal_weight=8e4, content_weight=2e4, style_weight=8e8, disp_weight=5e3)

    # train(dataset_path, style_image_path, model_dir, has_cuda, epochs=3, checkpoint_model_dir=checkpoint_model_dir,
    #       image_size=image_size, log_interval=100, checkpoint_interval=4000,
    #       model_filename="model_test_temp_8e4_content_2e4_style_8e8_disp_5e3_both_eyes",
    #       temporal_weight=8e4, content_weight=2e4, style_weight=2.3e8, disp_weight=5e3)


def show_pic_from_dataset():
    dataset_path = "../Data/Monkaa"
    train_dataset_path = os.path.join(dataset_path, "frames_cleanpass")
    flow_path = os.path.join(dataset_path, "optical_flow_resized")
    train_dataset = MyDataSet(train_dataset_path, flow_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    counter = 0
    for frames in tqdm(train_loader):
        (frames_curr, frames_next) = frames
        frame_curr_left = frames_curr[0]
        frame_curr_left = 255 * frame_curr_left.clone().clamp(0, 255).numpy()
        frame_left = frame_curr_left.transpose(2, 3, 1, 0).astype("uint8")
        frame_left = frame_left[:, :, :, 0]
        frame_left = Image.fromarray(frame_left)
        counter += 1
        if counter == 124:
            break
    frame_left.show()
    print(counter)

# Stylize:

def show_flow_after_style():
    model = "models/model_test_temp_1e5.pth"
    has_cuda = 1
    # img = Image.open("../Data/Monkaa/RGB_cleanpass/a_rain_of_stones_x2/left/0097.png")
    img = Image.open("../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png")
    left_frame_stylized = stylize(has_cuda, img, model)
    flow = utils_dataset.readFlow("../Data/Monkaa/optical_flow/eating_x2/left/OpticalFlowIntoFuture_0049_L.pfm")
    print("a")
    flow = flow[..., ::-1] - np.zeros_like(flow)
    flow = torch.from_numpy(flow)
    flow = flow.unsqueeze(0)
    left_frame_stylized = left_frame_stylized.unsqueeze(0)
    new_image, mask = utils.apply_flow(left_frame_stylized, flow)
    stylized_frame = new_image.clone().clamp(0, 255).numpy()
    stylized_frame = stylized_frame.astype("uint8")
    img = Image.fromarray(stylized_frame)
    img.show()

    # img_next = Image.open("../Data/Monkaa/frames_cleanpass/eating_x2/left/0050.png")
    # next_frame_stylized = stylize(has_cuda, img_next, model)
    # stylized_frame = next_frame_stylized.clone().clamp(0, 255).numpy()
    # stylized_frame = stylized_frame.astype("uint8")
    # img_next = Image.fromarray(stylized_frame)
    # img_next.show()


def show_stylized_image(img_path_left, img_path_right, model_path):
    has_cuda = 1
    img_left = Image.open(img_path_left)
    img_right = Image.open(img_path_right)
    left_frame_stylized, right_frame_stylized = stylize(has_cuda, img_left, img_right, model_path)

    stylized_frame_left = left_frame_stylized.clone().clamp(0, 255).cpu().numpy()
    stylized_frame_left = stylized_frame_left.transpose(1, 2, 0).astype("uint8")
    img_left = Image.fromarray(stylized_frame_left)
    stylized_frame_right = right_frame_stylized.clone().clamp(0, 255).cpu().numpy()
    stylized_frame_right = stylized_frame_right.transpose(1, 2, 0).astype("uint8")
    img_right = Image.fromarray(stylized_frame_right)
    img_left.show()
    img_right.show()


def show_flow_on_image(img_path, flow_path):
    # TODO: Optical flow doesn't work correctly (prob dimensions). Fix
    img = Image.open(img_path)

    transform_to_tensor = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    img = transform_to_tensor(img)
    C, H, W = img.shape
    im_batch = torch.ones((1, C, H, W))
    im_batch[0, :, :, :] = img
    img = im_batch

    flow = utils_dataset.readFlow(flow_path)
    flow = flow[..., ::] - np.zeros_like(flow)
    flow = torch.from_numpy(flow)
    flow = flow.unsqueeze(0)

    new_image, mask = utils.apply_flow(img, flow)
    new_image = np.asarray(255 * new_image).astype("uint8")
    mask = 255 * np.asarray(mask).astype("uint8")
    print((new_image.shape, type(new_image[0, 0, 0])))
    print(mask.shape, type(mask))
    Image.fromarray(new_image).show()
    Image.fromarray(mask).show()


def show_disparity_on_image(img_path, disparity_path):
    img = Image.open(img_path)

    transform_to_tensor = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    img = transform_to_tensor(img)
    C, H, W = img.shape
    im_batch = torch.ones((1, C, H, W))
    im_batch[0, :, :, :] = img
    img = im_batch

    disparity = utils_dataset.read(disparity_path)

    test_img = disparity / disparity.max() * 255
    disparity = disparity[..., ::] - np.zeros_like(disparity)
    disparity = torch.from_numpy(disparity)
    disparity = disparity.unsqueeze(0)
    # temp = disparity[:, :, :, None]
    # disparity = torch.cat((temp, torch.zeros_like(temp)), dim=3)

    new_image, mask = utils.apply_flow(img, disparity)
    new_image = np.asarray(255 * new_image).astype("uint8")
    mask = 255 * np.asarray(mask).astype("uint8")
    print((new_image.shape, type(new_image[0, 0, 0])))
    print(mask.shape, type(mask))
    Image.fromarray(new_image).show()
    # Image.fromarray(test_img).show()
    Image.fromarray(mask).show()


# train_models()

# img_path_left = "../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png"
# img_path_right = "../Data/Monkaa/frames_cleanpass/eating_x2/right/0049.png"

# img_path_left = "../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png"
# img_path_right = "../Data/Monkaa/frames_cleanpass/eating_x2/right/0049.png"

img_path_left = "images/content-images/cubic_left.jpg"
img_path_right = "images/content-images/cubic_right.jpg"

model_path = "models/model_test_temp_7.5e4_content_1e4_style_8e8_disp_5e3_both_eyes.pth"
show_stylized_image(img_path_left, img_path_right, model_path)

# img_path_left = "../Data/Monkaa/frames_cleanpass/treeflight_augmented1_x2/left/0358.png"
# disparity_path = "../Data/Monkaa/disparity_resized/treeflight_augmented1_x2/left/0358.flo"
#
# show_disparity_on_image(img_path_left,disparity_path)

