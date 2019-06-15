import torch
from PIL import Image
import fast_neural_style.neural_style.utils_dataset as utils_dataset
import numpy as np
import re


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
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


def apply_flow(img, flow_path):
    flow = utils_dataset.readFlow(flow_path)
    flow = np.round(flow)
    height, width, _ = np.asarray(img).shape

    new_pixel_place = np.indices((height, width)).transpose(1, 2, 0)
    new_pixel_place = new_pixel_place + flow[:, :, ::-1]

    new_pixel_place = new_pixel_place.astype(int)
    im_array = np.asarray(img)
    new_image = np.zeros_like(im_array)
    valid_indices = np.where((new_pixel_place[:, :, 0] >= 0) & (new_pixel_place[:, :, 0] < height) &
                             (new_pixel_place[:, :, 1] >= 0) & (new_pixel_place[:, :, 1] < width))
    new_pixel_place = new_pixel_place[valid_indices[0], valid_indices[1], :]
    new_image[new_pixel_place[:, 0], new_pixel_place[:, 1], :] = im_array[valid_indices[0], valid_indices[1], :]
    mask = np.zeros_like(img)
    mask[new_pixel_place[:, 0], new_pixel_place[:, 1]] = 1

    return new_image, mask


