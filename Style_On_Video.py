import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from fast_neural_style.neural_style.neural_style import stylize

cap = cv2.VideoCapture('Dinosaur.mp4')

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# model_name = "colors_temp_7e4_content_3e4_style_1e9_disp_1e5_epochs_6"
model_name = "model_test_temp_7.5e4_content_1e4_style_8e8_disp_5e3_both_eyes"
model = "./fast_neural_style/models/Mosaic/" + model_name + ".pth"
video_name = "Dinosaur_" + model_name + ".avi"
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
# cap.set(cv2.CAP_PROP_POS_FRAMES, 2300)

for i in tqdm(range(num_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    left_frame = frame[:, np.arange(0, width // 2).astype(int), :]
    right_frame = frame[:, np.arange(width // 2, width).astype(int), :]

    has_cuda = 1
    left_frame_stylized, right_frame_stylized = stylize(has_cuda, left_frame, right_frame, model)
    # stylized_frame = stylize(has_cuda, frame, model)

    stylized_frame = torch.cat((left_frame_stylized,right_frame_stylized), 2)  # Cat W dim
    stylized_frame = stylized_frame.clone().clamp(0, 255).cpu().numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    out.write(stylized_frame)

cap.release()
out.release()

