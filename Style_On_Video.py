import numpy as np
import cv2
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from fast_neural_style.neural_style.neural_style import stylize

cap = cv2.VideoCapture('Dinosaur.mp4')

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("Dinosaur_temp_0.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

for i in tqdm(range(num_frames // 7)):
    ret, frame = cap.read()
    if not ret:
        break

    left_frame = frame[:, np.arange(0, width // 2).astype(int), :]
    right_frame = frame[:, np.arange(width // 2, width).astype(int), :]

    has_cuda = 1
    model = "./fast_neural_style/models/model_test_temp_1e5.pth"
    left_frame_stylized = stylize(has_cuda, left_frame, model)
    right_frame_stylized = stylize(has_cuda, right_frame, model) # Shape after stylize: (C,H,W)

    stylized_frame = torch.cat((left_frame_stylized,right_frame_stylized), 2) # Cat W dim
    stylized_frame = stylized_frame.clone().clamp(0, 255).numpy()
    stylized_frame = stylized_frame.transpose(1, 2, 0).astype("uint8")
    out.write(stylized_frame)

cap.release()
out.release()

