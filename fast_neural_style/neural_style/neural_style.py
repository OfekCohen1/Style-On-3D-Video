import argparse
import os
import sys
import time
import re
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from fast_neural_style.neural_style.MyDataSet import MyDataSet
from PIL import Image
import fast_neural_style.neural_style.utils as utils
from fast_neural_style.neural_style.transformer_net import TransformerNet
from fast_neural_style.neural_style.vgg import Vgg16
import fast_neural_style.neural_style.losses as losses


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
          epochs=2, image_limit=None, checkpoint_model_dir=None, image_size=256, style_size=None, seed=42,
          content_weight=1e5, style_weight=1e10, temporal_weight=1e3,  # TODO: change temporal_weight default
          lr=1e-3, log_interval=500, checkpoint_interval=2000):
    device = torch.device("cuda" if has_cuda else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 1  # needs to be 1, batch is created using MyDataSet

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # videos_list = os.listdir(dataset_path)
    # train_dataset = {}
    # train_loader = {}
    # for video_name in videos_list:
    #     video_dataset_path = os.path.join(dataset_path, video_name)
    #     train_dataset[video_name] = MyDataSet(video_dataset_path, transform)
    #     train_loader[video_name] = DataLoader(train_dataset[video_name], batch_size=batch_size)

    # video_dataset_path = os.path.join(dataset_path, "Monkaa")  # dataset_path = "Data/Monkaa"
    train_dataset = MyDataSet(dataset_path, transform, image_limit=image_limit)  # remove if using all datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer_net = TransformerNet().to(device)
    optimizer = Adam(transformer_net.parameters(), lr)

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    style_image = utils.load_image(style_image_path, size=style_size)
    style_image = style_transform(style_image)
    style_image = style_image.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    for e in range(epochs):
        batch_num = 0
        # for video_name in videos_list:
        for frames_curr_next in tqdm(train_loader):
            (frames_curr, frames_next) = frames_curr_next
            batch_num += 1
            for i in [0, 1]:  # Left,  Right
                frame_curr = frames_curr[i]
                frame_next = frames_next[i]
                batch_size = len(frame_curr)
                optimizer.zero_grad()

                frame_curr = frame_curr.to(device)
                frame_style = transformer_net(frame_curr)

                features_frame = vgg(frame_curr)
                features_frame_style = vgg(frame_style)

                content_loss = losses.content_loss(features_frame, features_frame_style)
                style_loss = losses.style_loss(features_frame_style, gram_style, batch_size)
                flow_path = os.path.join(dataset_path, "optical_flow")
                temporal_loss = losses.temporal_loss(frame_curr, frame_next, flow_path, device)

                total_loss = (content_weight * content_loss + style_weight * style_loss
                              + temporal_weight * temporal_loss)
                total_loss.backward()
                optimizer.step()

                if (batch_num + 1) % log_interval == 0:  # TODO: Choose between TQDM and printing
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttemporal: {:.6f}" \
                           "\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, batch_num + 1, 2 * len(train_dataset),
                        content_loss.item(),
                        style_loss.item(),
                        temporal_loss.item(),
                        total_loss
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    output = utils.un_normalize_batch(output)

    return output[0]


def main():
    # TODO: This Doesn't Work. We changed image_path to image.
    # has_cuda = 1
    # content_image_path = "../images/content-images/ofek_garden.jpg"
    # output_image_path = "../images/output-images/ofek_garden-test.jpg"
    # model = "../models/mosaic.pth"
    # stylize(has_cuda, content_image_path, output_image_path, model)
    #
    image_size = 256
    dataset_path = "../../Data/Monkaa"
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = MyDataSet(dataset_path, transform)  # remove if using all datasets
    train_loader = DataLoader(train_dataset, batch_size=1)
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


if __name__ == "__main__":
    main()
