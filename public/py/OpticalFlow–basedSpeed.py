import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video path
video_path = 'D:/video/7月12日.mp4'

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Read first frame, convert to grayscale, downsample
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame")
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h, w = prev_gray.shape
scale = 0.25
prev_gray = cv2.resize(prev_gray, (int(w*scale), int(h*scale)))

speeds = []
times = []

step = 2  # sample every 2 frames
frame_idx = 0

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % step != 0:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (int(w*scale), int(h*scale)))
    
    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_small, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    speed = np.mean(mag) * (fps / step)
    speeds.append(speed)
    times.append(frame_idx / fps)
    
    prev_gray = gray_small

cap.release()

# Plot
plt.figure(figsize=(10, 4))
plt.plot(times, speeds, color='tab:orange', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Estimated Speed (optical flow)')
plt.title('Optical Flow–based Speed Waveform for the Video')
plt.grid(True)
plt.tight_layout()
plt.show()
