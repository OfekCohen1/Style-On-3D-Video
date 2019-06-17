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
import fast_neural_style.neural_style.losses as losses

# videos_list = os.listdir("../Data")
# videos_list = os.listdir(dataset_path)
# train_dataset = {}
# train_loader = {}
# for video_name in videos_list:
#     video_dataset_path = os.path.join(dataset_path, video_name)
#     train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
#     train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size=1)

image_size = 256
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])
def train_models():
    image_size = 256
    dataset_path = "../Data/Monkaa"
    style_image_path = "images/style-images/mosaic.jpg"
    model_dir = "../fast_neural_style/models/"
    checkpoint_model_dir = "../fast_neural_style/models/checkpoint_models"
    has_cuda = 1

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=200, checkpoint_interval=8000, model_filename="model_test_temp_0",
          temporal_weight=0, content_weight=1e5, style_weight=1e10)
    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=200, checkpoint_interval=8000, model_filename="model_test_temp_1e5",
          temporal_weight=1e5, content_weight=1e5, style_weight=1e10)
    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=200, checkpoint_interval=8000, model_filename="model_test_temp_1e7",
          temporal_weight=1e7,content_weight=1e5, style_weight=1e10)
    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=200, checkpoint_interval=8000, model_filename="model_test_temp_1e8",
          temporal_weight=1e8,content_weight=1e5, style_weight=1e10)

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

def show_stylized_image(img):
    model = "models/model_test_temp_1e5.pth"
    has_cuda = 1
    img = Image.open(img)
    left_frame_stylized = stylize(has_cuda, img, model)
    stylized_frame = left_frame_stylized.clone().clamp(0, 255).numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(stylized_frame)
    img.show()

def show_flow_on_image(img_path, flow_path):
    # TODO: Optical flow doesn't work correctly (prob dimensions). Fix
    img = Image.open(img_path)
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    img = transform_to_tensor(img)
    C, H, W = img.shape
    im_batch = torch.ones((1, C, H, W))
    im_batch[0, :, :, :] = img
    img = im_batch

    flow = utils_dataset.readFlow(flow_path)
    flow = flow[..., ::-1] - np.zeros_like(flow)
    flow = torch.from_numpy(flow)
    flow = flow.unsqueeze(0)

    new_image, mask = utils.apply_flow(img, flow)
    new_image = np.asarray(255 * new_image).astype("uint8")
    mask = 255 * np.asarray(mask).astype("uint8")
    print((new_image.shape, type(new_image[0, 0, 0])))
    print(mask.shape, type(mask))
    Image.fromarray(new_image).show()
    Image.fromarray(mask).show()


# img1 = "../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png"
# flow_path = "../Data/Monkaa/optical_flow/eating_x2/left/OpticalFlowIntoFuture_0049_L.pfm"
# # show_stylized_image(img1)
# img2 = "../Data/Monkaa/frames_cleanpass/eating_x2/left/0050.png"
# # show_stylized_image(img2)
#
# Image.open(img1).show()
# show_flow_on_image(img1, flow_path)
# Image.open(img2).show()


def resize_flow(flow, new_width, new_height):
    height, width, _ = flow.shape
    height_ratio = height / new_height
    width_ration = width / new_width
    x_axis = np.linspace(0, width - 1, new_width)
    y_axis = np.linspace(0, height - 1, new_height)
    x_axis = np.round(x_axis).astype(int)
    y_axis = np.round(y_axis).astype(int)
    xx, yy = np.meshgrid(x_axis, y_axis)
    flow = flow[yy, xx, :]
    flow[:, :, 0] = flow[:, :, 0] / width_ration
    flow[:, :, 1] = flow[:, :, 1] / height_ratio
    return flow


def show_optical_flow():
    # TODO: Optical flow doesn't work correctly (prob dimensions). Fix
    im = Image.open("../Data/Monkaa/RGB_cleanpass/left/0049.png")
    flow = utils_dataset.readFlow("../Data/Monkaa/optical_flow/forward/0049.pfm")
    # flow = utils_dataset.read("./flow_resize_test.flo")
    height, width, _ = np.asarray(im).shape

    ########### Flow Resize ###########

    # height = int(height / 2)
    # width = int(width / 2)
    height = 256
    width = 256
    im = im.resize((width, height), Image.ANTIALIAS)
    flow = resize_flow(flow, width, height)
    dir = "../test/new_folder"
    if not os.path.exists(dir):
        os.makedirs(dir)
    utils_dataset.write(os.path.join(dir, "flow_resize_test.flo"), flow)

    ###################################

    flow = np.round(flow)

    new_pixel_place = np.indices((height, width)).transpose(1, 2, 0)
    new_pixel_place = new_pixel_place+flow[:, :, ::-1]

    new_pixel_place = new_pixel_place.astype(int)
    im_array = np.asarray(im)
    new_image = np.zeros_like(im_array)
    valid_indices = np.where((new_pixel_place[:,:,0]>=0) & (new_pixel_place[:,:,0]<height)&
                             (new_pixel_place[:,:,1]>=0) & (new_pixel_place[:,:,1]<width))
    new_pixel_place = new_pixel_place[valid_indices[0], valid_indices[1],:]
    new_image[new_pixel_place[:,0], new_pixel_place[:,1], :] = im_array[valid_indices[0], valid_indices[1], :]
    mask = np.zeros_like(im)
    mask[new_pixel_place[:,0], new_pixel_place[:,1]] = 1;

    return new_image, mask


img = Image.open("../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png")
flow_path = "../Data/Monkaa/optical_flow/eating_x2/left/OpticalFlowIntoFuture_0049_L.pfm"
height = 256
width = 256
# img = img.resize((width, height), Image.ANTIALIAS)
transform_to_tensor = transforms.Compose([transforms.ToTensor()])
img = transform_to_tensor(img)
C, H, W = img.shape
im_batch = torch.ones((1, C, H, W))
im_batch[0, :, :, :] = img
img = im_batch
flow = utils_dataset.readFlow(flow_path)
flow = flow[..., ::-1] - np.zeros_like(flow)
flow = torch.from_numpy(flow)
flow = flow.unsqueeze(0)
new_image, mask = utils.apply_flow(img, flow)
new_image = np.asarray(255 * new_image).astype("uint8")
# new_image, mask = show_optical_flow()
Image.fromarray(new_image).show()
# Image.fromarray(mask*255).show()

Image.open("../Data/Monkaa/frames_cleanpass/eating_x2/left/0050.png").show()