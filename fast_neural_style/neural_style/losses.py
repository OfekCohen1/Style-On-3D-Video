import torch
import fast_neural_style.neural_style.utils as utils

def content_loss(features_frame, features_frame_style):
    mse_loss = torch.nn.MSELoss()
    content_loss = mse_loss(features_frame.relu2_2, features_frame_style.relu2_2)
    return content_loss


def style_loss(features_frame_style, gram_style, batch_size):
    mse_loss = torch.nn.MSELoss()
    style_loss = 0.
    for ft_frame_style, gm_s in zip(features_frame_style, gram_style):  # loop on feature layers
        gm_frame_style = utils.gram_matrix(ft_frame_style)
        style_loss += mse_loss(gm_frame_style, gm_s[:batch_size, :, :])

    return style_loss


def temporal_loss(frame_curr, frame_next, flow_path, device):
    frame_next_flow, mask = utils.apply_flow(frame_curr,flow_path)
    frame_next_flow = frame_next_flow.to(device)
    mask = mask.to(device)
    height, width= mask.shape
    temporal_loss = ((1/(height * width)) * torch.sum(mask * (frame_next_flow - frame_next)**2))
    return temporal_loss


