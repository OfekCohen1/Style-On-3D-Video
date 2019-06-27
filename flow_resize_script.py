import os
import numpy as np
import fast_neural_style.neural_style.utils_dataset as utils_dataset


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


og_dir = "./Data/Monkaa/optical_flow"  # TODO: Check for forward/ backward directories
new_dir = "./Data/Monkaa/optical_flow_resized"
new_width = 640
new_height = 360

for root, dirs, files in os.walk(og_dir):
    for file in files:
        if file.endswith(".pfm"):
            flow_path = os.path.join(root, file)
            flow = utils_dataset.readFlow(flow_path)
            flow_resized = resize_flow(flow, new_width, new_height)
            dir_folder, lr_folder = os.path.split(root)
            _, scene = os.path.split(dir_folder)
            flow_resized_path = os.path.join(new_dir, scene, lr_folder)
            if not os.path.exists(flow_resized_path):
                os.makedirs(flow_resized_path)
            file_name = file.replace('.pfm', '.flo')
            utils_dataset.write(os.path.join(flow_resized_path, file_name), flow_resized)

    # print(root)
    # print(dirs)
    # print(files)
    # print('--------------------------------')


