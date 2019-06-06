import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from fast_neural_style.neural_style.MyDataSet import MyDataSet

import fast_neural_style.neural_style.utils as utils
from fast_neural_style.neural_style.transformer_net import TransformerNet
from fast_neural_style.neural_style.vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(dataset_path, style_image_path, save_model_dir, has_cuda,
          epochs=2, batch_size=4, checkpoint_model_dir=None, image_size=256, style_size=None, seed=42,
          content_weight=1e5,
          style_weight=1e10, lr=1e-3, log_interval=500, checkpoint_interval=2000):
    device = torch.device("cuda" if has_cuda else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # TODO: Change Normalization to ImageNet Norm
    ])
    videos_list = os.listdir(dataset_path)
    train_dataset = {}
    train_loader = {}
    for video_name in videos_list:
        video_dataset_path = os.path.join(dataset_path, video_name)
        train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
        train_loader = DataLoader(train_dataset[video_name], batch_size=batch_size)

    transformer_net = TransformerNet().to(device)
    optimizer = Adam(transformer_net.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # TODO: Change Normalization to ImageNet Norm
    ])
    style = utils.load_image(style_image_path, size=style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(epochs):
        transformer_net.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for video_name in videos_list:
            for batch_num, both_frames in enumerate(train_loader[video_name]):
                for frame in both_frames:
                    bath_size = len(frame)
                    count += bath_size
                    optimizer.zero_grad()

                    frame = frame.to(device)
                    frame_style = transformer_net(frame)

                    frame_style = utils.normalize_batch(frame_style)
                    frame = utils.normalize_batch(frame)

                    features_frame = vgg(frame)
                    features_frame_style = vgg(frame_style)

                    content_loss = content_weight * mse_loss(features_frame.relu2_2, features_frame_style.relu2_2)

                    style_loss = 0.
                    for ft_frame_style, gm_s in zip(features_frame_style, gram_style):  # loop on feature layers
                        gm_frame_style = utils.gram_matrix(ft_frame_style)
                        style_loss += mse_loss(gm_frame_style, gm_s[:bath_size, :, :])
                    style_loss *= style_weight

                    total_loss = content_loss + style_loss
                    total_loss.backward()
                    optimizer.step()

                    agg_content_loss += content_loss.item()
                    agg_style_loss += style_loss.item()

                    if (batch_num + 1) % log_interval == 0:
                        mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, 2 * len(train_dataset),
                            content_loss.item(),
                            style_loss.item(),
                            (content_loss.item() + style_loss.item())
                        )
                        print(mesg)

                    if checkpoint_model_dir is not None and (batch_num + 1) % checkpoint_interval == 0:
                        transformer_net.eval().cpu()
                        ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_num + 1) + ".pth"
                        ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                        torch.save(transformer_net.state_dict(), ckpt_model_path)
                        transformer_net.to(device).train()

    # save model
    transformer_net.eval().cpu()
    # save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
    #     content_weight) + "_" + str(style_weight) + ".model"
    save_model_filename = "myModel.pth"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    torch.save(transformer_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(has_cuda, content_image, model, output_image_path=None, content_scale=None):
    device = torch.device("cuda" if has_cuda else "cpu")

    # content_image = utils.load_image(content_image_path, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    # utils.save_image(output_image_path, output[0])
    return output[0]


def main():
    # TODO: This Doesn't Work. We changed image_path to image.
    has_cuda = 1
    content_image_path = "../images/content-images/ofek_garden.jpg"
    output_image_path = "../images/output-images/ofek_garden-test.jpg"
    model = "../models/mosaic.pth"
    stylize(has_cuda, content_image_path, output_image_path, model)


if __name__ == "__main__":
    main()
