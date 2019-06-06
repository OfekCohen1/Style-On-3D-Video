from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataSet(Dataset):
    """ Expects different videos to be in different files, ie: "Data/VideoName/Frame01" """
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.frame_path_left = os.path.join(self.data_path, "RGB_cleanpass", "left")
        self.id_list_left = os.listdir(self.frame_path_left)
        self.frame_path_right = os.path.join(self.data_path, "RGB_cleanpass", "right")
        self.id_list_right = os.listdir(self.frame_path_right)
        self.transform = transform

    def __getitem__(self, num_frame):
        frame_path_left = os.path.join(self.frame_path_left, self.id_list_left[num_frame])
        frame_path_right = os.path.join(self.frame_path_right, self.id_list_right[num_frame])
        frame_left = Image.open(frame_path_left)
        frame_right = Image.open(frame_path_right)
        frame_left = self.transform(frame_left)
        frame_right = self.transform(frame_right)

        return frame_left, frame_right

    def __len__(self):
        return len(self.id_list_left)
