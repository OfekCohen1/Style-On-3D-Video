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
    # train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
    #       image_size=image_size, log_interval=100, checkpoint_interval=8000,
    #       model_filename="model_test_temp_1e9_style_1e14_content_1e9_epochs_4",
    #       temporal_weight=1e9, content_weight=1e9, style_weight=1e14)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=8000,
          model_filename="model_test_temp_7e9_style_1e14_content_1e9_epochs_4",
          temporal_weight=7e9, content_weight=1e9, style_weight=1e14)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=8000,
          model_filename="model_test_temp_8e9_style_1e14_content_1e9_epochs_4",
          temporal_weight=8e9, content_weight=1e9, style_weight=1e14)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=8000,
          model_filename="model_test_temp_9e9_style_1e14_content_1e9_epochs_4",
          temporal_weight=9e9, content_weight=1e9, style_weight=1e14)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=8000,
          model_filename="model_test_temp_10e9_style_1e14_content_1e9_epochs_4",
          temporal_weight=1e10, content_weight=1e9, style_weight=1e14)

    train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
          image_size=image_size, log_interval=100, checkpoint_interval=8000,
          model_filename="model_test_temp_12e9_style_1e14_content_1e9_epochs_4",
          temporal_weight=12e10, content_weight=1e9, style_weight=1e14)
    # train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
    #       image_size=image_size, log_interval=100, checkpoint_interval=8000,
    #       model_filename="model_test_temp_1e9_style_1e14_content_5e9_epochs_4",
    #       temporal_weight=1e9, content_weight=5e9, style_weight=1e14)
    #
    # train(dataset_path, style_image_path, model_dir, has_cuda, epochs=4, checkpoint_model_dir=checkpoint_model_dir,
    #       image_size=image_size, log_interval=100, checkpoint_interval=8000,
    #       model_filename="model_test_temp_1e9_style_5e14_content_1e9_epochs_4",
    #       temporal_weight=1e9, content_weight=1e9, style_weight=5e14)

    # model_init = "models/model_test_temp_5e5.pth"
    # train(dataset_path, style_image_path, model_dir, has_cuda, epochs=1, checkpoint_model_dir=checkpoint_model_dir,
    #       image_size=image_size, log_interval=1, checkpoint_interval=2, model_filename="model_test_short_5e6",
    #       temporal_weight=5e6, content_weight=1e5, style_weight=1e10, image_limit=100, model_init=model_init)


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

def show_stylized_image(img_path, model_path):
    has_cuda = 1
    img = Image.open(img_path)
    left_frame_stylized = stylize(has_cuda, img, model_path)
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


# img1_path = "../Data/Monkaa/frames_cleanpass/eating_x2/left/0049.png"
# img2_path = "images/content-images/amber.jpg"
# flow_path = "../Data/Monkaa/optical_flow/a_rain_of_stones_x2/right/OpticalFlowIntoFuture_0046_R.pfm"
# model_path = "models/model_test_temp_7e9_style_1e14_content_1e9_epochs_4.pth"
# model_path = "models/checkpoint_models/model_test_temp_1e9_style_1e14_content_1e9_epochs_4_ckpt_epoch_4_batch_id_8000.pth"

# show_stylized_image(img1, model_path)
# img2 = "../Data/Monkaa/frames_cleanpass/a_rain_of_stones_x2/right/0046.png"
# show_stylized_image(img2_path, model_path)
# train_models()
# show_flow_on_image(img2, flow_path)

# img_cubic = Image.open("images/content-images/cubic.jpg")
# img_cubic = np.array(img_cubic)
# cubic_left = Image.fromarray(img_cubic[:,0:1024,:])
# cubic_right = Image.fromarray(img_cubic[:,1024:2048,:])
# cubic_left.save("images/content-images/cubic_left.jpg")
# cubic_right.save("images/content-images/cubic_right.jpg")
