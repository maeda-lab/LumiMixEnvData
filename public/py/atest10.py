import cv2, numpy as np
import matplotlib.pyplot as plt

vid = r"D:\vectionProject\public\freq_test_videos\highpass_dog_1.2_4_gain10.mp4"  # 改成你要分析的
cap = cv2.VideoCapture(vid)

vals = []
prev = None
while True:
    ok, fr = cap.read()
    if not ok: break
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if prev is None:
        prev = gray
        continue
    # 简单的“变化能量”：帧间差（反映重影/变化程度）
    vals.append(float(np.mean(np.abs(gray - prev))))
    prev = gray

cap.release()
print("mean:", np.mean(vals), "max:", np.max(vals))

plt.plot(vals)
plt.title("Frame-to-frame change energy (proxy for ghosting/motion strength)")
plt.xlabel("frame")
plt.ylabel("mean abs diff")
plt.show()
