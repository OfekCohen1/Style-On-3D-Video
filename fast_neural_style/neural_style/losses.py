import torch
import fast_neural_style.neural_style.utils as utils
from PIL import Image


def content_loss(features_frame, features_frame_style):
    mse_loss = torch.nn.MSELoss()
    cont_loss = mse_loss(features_frame.relu2_2, features_frame_style.relu2_2)
    return cont_loss


def style_loss(features_frame_style, gram_style, batch_size):
    mse_loss = torch.nn.MSELoss()
    style_loss = 0
    for ft_frame_style, gm_s in zip(features_frame_style, gram_style):  # loop on feature layers
        gm_frame_style = utils.gram_matrix(ft_frame_style)
        style_loss += mse_loss(gm_frame_style, gm_s[:batch_size, :, :])

    return style_loss


def temporal_loss(frame_curr, frame_next, flow, device, to_save=False, batch_num=None, e=None):
    frame_next_flow, mask = utils.apply_flow(frame_curr, flow)
    frame_next_flow = frame_next_flow.to(device)
    mask = mask.to(device)
    frame_next = frame_next.permute(2, 3, 1, 0)
    frame_next = frame_next[:, :, :, 0]  # H x W x C
    frame_curr = frame_curr.permute(2, 3, 1, 0).squeeze(3)
    height, width, _ = mask.shape

    if to_save:
        namefile_frame_curr = 'test_images/frame_curr_style/frame_curr_style_epo' +\
                              str(e) + 'batch_num' + str(batch_num) + '.png'
        namefile_frame_next = 'test_images/frame_next_style/frame_next_style_epo_' +\
                              str(e) + 'batch_num_' + str(batch_num) + '.png'
        namefile_frame_next_flow = ('test_images/frame_flow_style/frame_next_flow_style_epo' + str(e) +
                                    'batch_num_' + str(batch_num) + '.png')
        namefile_frame_next_flow_mask = ('test_images/frame_flow_mask/frame_next_flow_mask_epo' + str(e) +
                                    'batch_num_' + str(batch_num) + '.png')
        utils.save_image_loss(frame_curr, namefile_frame_curr)
        utils.save_image_loss(frame_next, namefile_frame_next)
        utils.save_image_loss(frame_next_flow, namefile_frame_next_flow)
        utils.save_image_loss_mask(mask, namefile_frame_next_flow_mask)

        # utils.save_image_loss(mask, namefile_frame_next_flow_mask)

    temporal_loss = ((1 / (height * width)) * torch.sum(mask * (frame_next_flow - frame_next) ** 2))
    return temporal_loss


def tv_loss(frame_curr):
    total_variation = (torch.sum(torch.abs(frame_curr[:, :, :, :-1] - frame_curr[:, :, :, 1:])) +
                       torch.sum(torch.abs(frame_curr[:, :, :-1, :] - frame_curr[:, :, 1:, :])))

    return total_variation
