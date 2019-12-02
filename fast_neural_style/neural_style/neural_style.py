import argparse
import os
import sys
import time
import re

from cv2 import text_OCRBeamSearchDecoder
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
          epochs=2, image_limit=None, checkpoint_model_dir=None, image_size=(360, 640), style_size=None, seed=42,
          content_weight=1, style_weight=10, temporal_weight=10, tv_weight=1e-3, disp_weight=1e-3, lr=1e-3,
          log_interval=500, checkpoint_interval=2000, model_filename="myModel", model_init=None):
    device = torch.device("cuda" if has_cuda else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 1  # needs to be 1, batch is created using MyDataSet
    loss_list = []
    loss_filename = model_filename + '_losses.txt'

    transform = transforms.Compose([
        transforms.Resize(image_size),
        # transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
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

    train_dataset = MyDataSet(dataset_path, transform,
                              image_limit=image_limit)  # remove if using all datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    if model_init is not None:
        transformer_net = TransformerNet()
        state_dict = torch.load(model_init)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        # for k in list(state_dict.keys()):
        #     if re.search(r'in\d+\.running_(mean|var)$', k):
        #         del state_dict[k]
        transformer_net.load_state_dict(state_dict)
        transformer_net.to(device)
    else:
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
        for frames_and_flow in tqdm(train_loader):
            (frames_curr_lr, flow_lr, frames_next_lr, disparity) = frames_and_flow
            batch_num += 1
            total_loss = 0

            to_save = (batch_num + 1) % checkpoint_interval == 0
            optimizer.zero_grad()

            frames_left_batch = torch.cat((frames_curr_lr[0], frames_next_lr[0]), 0)
            frames_right_batch = torch.cat((frames_curr_lr[1], frames_next_lr[1]), 0)

            frames_left_batch = frames_left_batch.to(device)
            frames_right_batch = frames_right_batch.to(device)

            frame_style_left, frame_style_right = transformer_net(frames_left_batch, frames_right_batch)  # Two batches 2 x 3 x H x W
            frame_curr_style_combined = (frame_style_left[0, ::].unsqueeze(0), frame_style_right[0, ::].unsqueeze(0))
            frame_next_style_combined = (frame_style_left[1, ::].unsqueeze(0), frame_style_right[1, ::].unsqueeze(0))

            disparity_loss_l2r = losses.disparity_loss(frame_curr_style_combined[0], frame_curr_style_combined[1],
                                                       disparity[0], device, to_save, batch_num, e)
            disparity_loss_r2l = losses.disparity_loss(frame_curr_style_combined[1], frame_curr_style_combined[0],
                                                       disparity[1], device)
            total_loss = disp_weight * (disparity_loss_l2r + disparity_loss_r2l)
            # total_loss = disp_weight * disparity_loss_l2r

            for i in [0, 1]:  # Left,  Right
                frame_curr = frames_curr_lr[i]
                frame_next = frames_next_lr[i]
                frame_curr = frame_curr.to(device)
                frame_next = frame_next.to(device)
                flow = flow_lr[i]
                batch_size = len(frame_curr)
                frame_style = frame_curr_style_combined[i]
                frame_next_style = frame_next_style_combined[i]
                features_frame = vgg(frame_curr)
                features_frame_style = vgg(frame_style)
                content_loss = losses.content_loss(features_frame, features_frame_style)
                style_loss = losses.style_loss(features_frame_style, gram_style, batch_size)
                if to_save:
                    temporal_loss = losses.temporal_loss(frame_style, frame_next_style,
                                                         flow, device, to_save=to_save, batch_num=batch_num, e=e)
                else:
                    temporal_loss = losses.temporal_loss(frame_style, frame_next_style, flow, device)
                tv_loss = losses.tv_loss(frame_curr)
                total_loss = total_loss + (content_weight * content_loss + style_weight * style_loss
                                           + temporal_weight * temporal_loss)

                # Save stuff:
                frame_curr_to_save = frame_curr.permute(2, 3, 1, 0).squeeze(3)
                frame_next_to_save = frame_next.permute(2, 3, 1, 0).squeeze(3)
                namefile_frame_curr = 'test_images/frame_curr/frame_curr_epo' + str(e) + 'batch_num' + str(
                    batch_num) + "eye" + str(i) + '.png'
                namefile_frame_next = 'test_images/frame_next/frame_next_epo' + str(e) + 'batch_num' + str(
                    batch_num) + "eye" + str(i) + '.png'
                namefile_frame_flow = 'test_images/frame_flow/frame_next_epo' + str(e) + 'batch_num' + str(
                    batch_num) + "eye" + str(i) + '.png'
                frame_flow, _ = utils.apply_flow(frame_curr, flow)
                if to_save:
                    utils.save_image_loss(frame_curr_to_save, namefile_frame_curr)
                    utils.save_image_loss(frame_next_to_save, namefile_frame_next)
                    utils.save_image_loss(frame_flow, namefile_frame_flow)

            # After we ran on both eyes
            total_loss.backward()
            optimizer.step()

            if (batch_num + 1) % log_interval == 0:  # TODO: Choose between TQDM and printing
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttemporal: {:.6f}" \
                       "\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, batch_num + 1, len(train_dataset),
                    content_loss.item(),
                    style_loss.item(),
                    temporal_loss.item(),
                    total_loss.item()
                )
                # print(mesg)
                losses_string = (str(content_loss.item()) + "," + str(style_loss.item()) + "," +
                                 str(temporal_loss.item()) + "," + str(total_loss.item()))
                loss_list.append(losses_string)
                utils.save_loss_file(loss_list, loss_filename)

            if checkpoint_model_dir is not None and (batch_num + 1) % checkpoint_interval == 0:
                transformer_net.eval().cpu()
                ckpt_model_filename = (model_filename + "_ckpt_epoch_" +
                                       str(e + 1) + "_batch_id_" + str(batch_num + 1) + ".pth")
                ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer_net.state_dict(), ckpt_model_path)
                transformer_net.to(device).train()

    # save model
    transformer_net.eval().cpu()
    # save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
    #     content_weight) + "_" + str(style_weight) + ".model"
    save_model_path = os.path.join(save_model_dir, model_filename + ".pth")
    torch.save(transformer_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(has_cuda, left_image, right_image, model, output_image_path=None, content_scale=None):
    device = torch.device("cuda" if has_cuda else "cpu")

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    left_image = content_transform(left_image)
    left_image = left_image.unsqueeze(0).to(device)
    right_image = content_transform(right_image)
    right_image = right_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output_left, output_right = style_model(left_image, right_image)

    # output_left = output_left.cpu()
    # output_right = output_right.cpu()
    output_left = utils.un_normalize_batch(output_left)
    output_right = utils.un_normalize_batch(output_right)

    return output_left[0], output_right[0]


def main():
    # TODO: This Doesn't Work. We changed image_path to image.
    has_cuda = 1
    # content_image_path = "../images/content-images/ofek_garden.jpg"
    # output_image_path = "../images/output-images/ofek_garden-test.jpg"
    # model = "../models/mosaic.pth"
    # stylize(has_cuda, content_image_path, output_image_path, model)
    #
    # image_size = 256
    # dataset_path = "../../Data/Monkaa"
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # train_dataset = MyDataSet(dataset_path, transform)  # remove if using all datasets
    # train_loader = DataLoader(train_dataset, batch_size=1)
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


if __name__ == "__main__":
    main()
