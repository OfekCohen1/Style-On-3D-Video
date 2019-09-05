import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from fast_neural_style.neural_style.neural_style import stylize

cap = cv2.VideoCapture('Elephants.mov')

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("Elephants_style_temp_7e9.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
# cap.set(cv2.CAP_PROP_POS_FRAMES, 2300)

for i in tqdm(range(num_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # left_frame = frame[:, np.arange(0, width // 2).astype(int), :]
    # right_frame = frame[:, np.arange(width // 2, width).astype(int), :]

    has_cuda = 1
    model = "./fast_neural_style/models/model_test_temp_7e9.pth"
    # left_frame_stylized = stylize(has_cuda, left_frame, model)
    # right_frame_stylized = stylize(has_cuda, right_frame, model) # Shape after stylize: (C,H,W)
    stylized_frame = stylize(has_cuda, frame, model)

    # stylized_frame = torch.cat((left_frame_stylized,right_frame_stylized), 2) # Cat W dim
    stylized_frame = stylized_frame.clone().clamp(0, 255).numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    out.write(stylized_frame)

cap.release()
out.release()

