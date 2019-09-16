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


def resize_flow_script():
    og_dir = "./Data/Monkaa/optical_flow"
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


def fix_disparity(disparity):
    height, width = disparity.shape
    temp = np.zeros_like(disparity) - 1  # holds original column number of new image created by applying disparity
    disparity_fixed = np.round(disparity).astype(int)
    OUT_OF_BOUNDS = 5000
    for row in range(height - 1):
        for col in range(width - 1):
            new_col = col + disparity_fixed[row, col]
            if (new_col >= 0) & (new_col < width):  # pixel who move outside of the image will be ignored
                if temp[row, new_col] == -1:  # check if a pixel already exists in new place
                    temp[row, new_col] = col  # if not then sets there the current pixel
                else:
                    if np.abs(disparity_fixed[row, col]) > np.abs(new_col - temp[row, new_col]):
                        # if such a pixel exists, checks if it moved more then the current pixel by checking
                        # the distance between the original column stored in temp and the new column
                        disparity_fixed[row, int(temp[row, new_col])] = width + OUT_OF_BOUNDS      # if it did then change the disparity value
                                                                            # of the already existing pixel to outside
                                                                            # the bounds of the image
                        temp[row, new_col] = col    # set the new original column of the current pixel in temp
                    else:
                        disparity_fixed[row, col] = width + OUT_OF_BOUNDS     # if the new current pixel moved less then the already
                                                            # existing pixel then change the disparity  of the current
                                                            # pixel to outside the bounds of the image
    return disparity_fixed


def fix_disparity_script():
    og_dir = "./Data/Monkaa/disparity"
    new_dir = "./Data/Monkaa/disparity_fixed"

    for root, dirs, files in os.walk(og_dir):
        for file in files:
            if file.endswith(".pfm"):
                disparity_path = os.path.join(root, file)

                dir_folder, lr_folder = os.path.split(root)
                _, scene = os.path.split(dir_folder)

                disparity = utils_dataset.read(disparity_path)

                if lr_folder == "left":
                    disparity = disparity * (-1)
                disparity_fixed = fix_disparity(disparity)

                disparity_fixed = disparity_fixed[:, :, None]   # add empty y coordinates
                temp = np.zeros_like(disparity_fixed)
                disparity_fixed = np.concatenate([disparity_fixed, temp], axis=2)

                disparity_fixed_path = os.path.join(new_dir, scene, lr_folder)
                if not os.path.exists(disparity_fixed_path):
                    os.makedirs(disparity_fixed_path)
                file_name = file.replace('.pfm', '.flo')
                utils_dataset.write(os.path.join(disparity_fixed_path, file_name), disparity_fixed)


fix_disparity_script()
