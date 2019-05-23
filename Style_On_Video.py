import numpy as np
import cv2
from tqdm import tqdm

cap = cv2.VideoCapture('Dinosaur.mp4')

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("Half_Dinosaur.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

for i in tqdm(range(num_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    frame[:, np.arange(width / 2, width).astype(int), :] = 0
    out.write(frame)

    # cv2.imshow('frame', frame)

cap.release()
out.release()
print([num_frames, width, height, fps])
