import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 读取两张图像 ==========
# 把这里改成你的实际路径
img1_path = "capture1_001.png"
img2_path = "capture1_061.png"

# 以 BGR 读入
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise RuntimeError("图像读取失败，请检查路径和文件名。")

if img1.shape != img2.shape:
    raise RuntimeError("两张图像尺寸不一样，无法直接混合。")

h, w, _ = img1.shape
print("Image size:", w, "x", h)

# 转为 float32，归一化到 0~1
img1_f = img1.astype(np.float32) / 255.0
img2_f = img2.astype(np.float32) / 255.0

# ========== 2. 设定 ROI（树干所在区域） ==========
# 请根据实际树的位置调整这几个值：
# x0, y0: ROI 左上角坐标
# roi_w, roi_h: ROI 宽和高
x0 = 1364  # 例子值：你要根据自己的图调整
y0 = 252
roi_w = 333
roi_h = 305

# 确保 ROI 在图像范围内
x0 = max(0, min(x0, w-1))
y0 = max(0, min(y0, h-1))
roi_w = min(roi_w, w - x0)
roi_h = min(roi_h, h - y0)
print("ROI:", x0, y0, roi_w, roi_h)

# ========== 3. 对一系列 alpha 计算“树干位置” ==========
alphas = np.linspace(0.0, 1.0, 21)  # 21 个点，从 0 到 1，每隔 0.05
x_positions = []

for alpha in alphas:
    # 3.1 混合图像
    blend = (1.0 - alpha) * img1_f + alpha * img2_f

    # 3.2 取 ROI
    roi = blend[y0:y0+roi_h, x0:x0+roi_w, :]

    # 3.3 转灰度
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 3.4 Sobel 求水平梯度 gx
    #   dx=1, dy=0 表示对 x 求导；ksize=3 表示 3x3 卷积核
    gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    mgx = np.abs(gx)

    # 3.5 沿 y 方向把每一列的梯度强度累加，得到 1D profile
    #    profile[x] 越大，说明这一列有越强的竖直边缘
    profile = mgx.sum(axis=0)  # shape: (roi_w,)

    # 3.6 找到 profile 的最大值位置（树干边缘）
    x_peak_local = int(np.argmax(profile))  # ROI 内的 x 索引
    x_peak_global = x0 + x_peak_local      # 转为整图坐标

    x_positions.append(x_peak_global)

    print(f"alpha={alpha:.2f}, x_local={x_peak_local}, x_global={x_peak_global}")

x_positions = np.array(x_positions)

# ========== 4. 画出 alpha vs x_position 的曲线 ==========
plt.figure(figsize=(6, 4))
plt.plot(alphas, x_positions, marker='o')
plt.xlabel("alpha (luminance blend ratio)")
plt.ylabel("tree edge X position (pixels)")
plt.title("Perceived tree position vs alpha")
plt.grid(True)
plt.tight_layout()
plt.show()

# 如果你想保存数据：
out = np.stack([alphas, x_positions], axis=1)
np.savetxt("tree_position_vs_alpha.csv", out, delimiter=",", header="alpha,x_position", comments='')
print("保存到 tree_position_vs_alpha.csv")
