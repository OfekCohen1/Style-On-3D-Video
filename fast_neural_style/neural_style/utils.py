import torch
from PIL import Image
import fast_neural_style.neural_style.utils_dataset as utils_dataset
import numpy as np
import re
# import matplotlib.pyplot as plt

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):  # TODO: Remove this save_image and change with save_image_loss
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def un_normalize_batch(batch):
    # un- normalize imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return ((batch * std) + mean) * 255


def apply_flow(img, flow):
    """ Flow is Tensor B x H x W x C , Image is Tensor B x C x H x W """
    flow = flow[0, :, :, :]  # H x W x C
    height, width, _ = flow.shape
    img = img.permute(2, 3, 1, 0)
    img = img[:, :, :, 0]  # H x W x C
    img_np = img.clone().cpu().detach().numpy()  # Image input is as a tensor, but apply_flow uses numpy
    flow = np.asarray(flow.cpu())
    flow = np.round(flow)

    new_pixel_place = np.indices((height, width)).transpose(1, 2, 0)
    new_pixel_place = new_pixel_place + flow[:, :, ::-1]

    new_pixel_place = new_pixel_place.astype(int)
    im_array = np.asarray(img_np)
    new_image = np.zeros_like(im_array)

    valid_indices = np.where((new_pixel_place[:, :, 0] >= 0) & (new_pixel_place[:, :, 0] < height) &
                             (new_pixel_place[:, :, 1] >= 0) & (new_pixel_place[:, :, 1] < width))
    new_pixel_place = new_pixel_place[valid_indices[0], valid_indices[1], :]
    new_image[new_pixel_place[:, 0], new_pixel_place[:, 1], :] = im_array[valid_indices[0], valid_indices[1], :]

    # for row in range(height-1, 0, -1):
    #     for col in range(width-1, 0, -1):
    # for row in range(height - 1):
    #     for col in range(width - 1):
    #         new_row = new_pixel_place[row, col, 0]
    #         new_col = new_pixel_place[row, col, 1]
    #         if (new_row >= 0) & (new_row < height) & (new_col >= 0) & (new_col < width):
    #             new_image[new_row, new_col, :] = im_array[row, col, :]

    mask = np.zeros_like(img_np)
    mask[new_pixel_place[:, 0], new_pixel_place[:, 1]] = 1

    new_image = torch.as_tensor(new_image)
    mask = torch.as_tensor(mask)
    return new_image, mask


def save_loss_file(loss_list, file_path):
    f = open(file_path, "w+")
    for item in loss_list:
        f.write("%s\n" % item)
    f.close()


# def read_loss_file(filename):
#     f = open(filename, 'r')
#     loss_list = []
#     contents = f.readlines()
#     for item in contents:
#         loss_list.append(float(item))
#
#     plt.plot(loss_list)
#     return loss_list

def save_image_loss(frame, name_file):
    """ Image received is H x W x C Tensor"""
    mean = frame.new_tensor([0.485, 0.456, 0.406]).view(1, 1, -1)
    std = frame.new_tensor([0.229, 0.224, 0.225]).view(1, 1, -1)
    frame = ((frame * std) + mean) * 255
    frame_numpy = frame.clone().detach().cpu().numpy().astype("uint8")
    frame_image = Image.fromarray(frame_numpy)
    frame_image.save(name_file)


def save_image_loss_mask(mask, name_file):
    mask = mask.clone().detach().cpu()
    mask = 255 * np.asarray(mask).astype("uint8")
    frame_image = Image.fromarray(mask)
    frame_image.save(name_file)
