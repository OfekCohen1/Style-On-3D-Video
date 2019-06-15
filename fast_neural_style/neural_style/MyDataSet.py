from torch.utils.data import Dataset
import os
from PIL import Image
import fast_neural_style.neural_style.utils_dataset as utils_dataset


class MyDataSet(Dataset):
    """ Expects different videos to be in different files, ie: "Data/VideoName/Frame01" """  # TODO: Fix this doc
    def __init__(self, train_dataset_path, flow_path, transform, image_limit = None):
        self.flow_path = flow_path
        # self.frame_path_left = os.path.join(self.data_path, "RGB_cleanpass", "left")
        # self.id_list_left = os.listdir(self.frame_path_left)
        # self.frame_path_right = os.path.join(self.data_path, "RGB_cleanpass", "right")
        # self.id_list_right = os.listdir(self.frame_path_right)
        self.transform = transform
        self.pic_direcs_list = utils_dataset.get_pics_direcs(train_dataset_path, image_limit=image_limit)

    def __getitem__(self, num_frame):
        """ Returns (X_curr_left, X_curr_right), (X_flow_left, X_flow_right), (X_next_left, X_next_right)"""
        frame_path_curr_left = self.pic_direcs_list[num_frame][0]
        frame_path_curr_right = self.pic_direcs_list[num_frame][1]
        frame_curr_left = Image.open(frame_path_curr_left)
        frame_curr_right = Image.open(frame_path_curr_right)
        frame_curr_left = self.transform(frame_curr_left)
        frame_curr_right = self.transform(frame_curr_right)

        # Get directory for next picture
        frame_name_num = os.path.basename(frame_path_curr_left)  # ie "0001.png"
        suffix = os.path.splitext(frame_path_curr_left)[1]  # .png
        direc_without_num_frame_left = frame_path_curr_left.replace(frame_name_num, '')
        direc_without_num_frame_right = frame_path_curr_right.replace(frame_name_num, '')
        frame_name_string = frame_name_num.replace(suffix, '')  # 0001

        num_digits = len(frame_name_string)  # 0001 -> num_digits = 4
        int_curr = int(frame_name_string)
        int_next = int_curr + 1  # n+1 as int
        string_next = str(int_next)
        num_digits_next = len(string_next)
        string_next = '0' * (num_digits - num_digits_next) + string_next + suffix  # 0002.png

        frame_path_next_left = os.path.join(direc_without_num_frame_left, string_next)
        frame_path_next_right = os.path.join(direc_without_num_frame_right, string_next)
        frame_next_left = Image.open(frame_path_next_left)
        frame_next_right = Image.open(frame_path_next_right)
        frame_next_left = self.transform(frame_next_left)
        frame_next_right = self.transform(frame_next_right)

        # Get directory for flow of current picture
        direc_without_num_frame_left = os.path.normpath(direc_without_num_frame_left)
        direc_without_left = os.path.split(direc_without_num_frame_left)[0]
        scene_name = os.path.split(direc_without_left)[1]  # ie: a_rain_of_stones_x2
        flow_name_num_left = "OpticalFlowIntoFuture_" + frame_name_string + "_L.flo"  # OpticalFlowIntoFuture_0000_L.flo
        flow_name_num_right = "OpticalFlowIntoFuture_" + frame_name_string + "_R.flo" # OpticalFlowIntoFuture_0000_L.flo
        flow_path_left = os.path.join(self.flow_path, scene_name, "left", flow_name_num_left)
        flow_path_right = os.path.join(self.flow_path, scene_name, "right", flow_name_num_right)
        flow_left = utils_dataset.readFlow(flow_path_left)
        flow_right = utils_dataset.readFlow(flow_path_right)

        return (frame_curr_left, frame_curr_right), (flow_left, flow_right), (frame_next_left, frame_next_right)

    def __len__(self):
        return len(self.pic_direcs_list)

