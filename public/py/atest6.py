import cv2, numpy as np, imageio.v2 as imageio

lin   = r"D:\vectionProject\public\bandpass_videos\band_hi_lin.mp4"
phase = r"D:\vectionProject\public\bandpass_videos\band_hi_phase.mp4"
out   = r"D:\vectionProject\public\bandpass_videos\diff_lin_vs_phase_x20.mp4"

capA = cv2.VideoCapture(lin)
capB = cv2.VideoCapture(phase)
fps = capA.get(cv2.CAP_PROP_FPS)

writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)

scale = 20.0   # 放大倍数：20~30都可以试
try:
    while True:
        ok1, a = capA.read()
        ok2, b = capB.read()
        if not ok1 or not ok2: break

        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        diff = np.clip(0.5 + (b - a) * scale, 0, 1)  # 0.5为中灰，差异正负都可见
        writer.append_data((diff*255+0.5).astype(np.uint8))
finally:
    capA.release(); capB.release(); writer.close()

print("saved:", out)
