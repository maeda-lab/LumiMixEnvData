import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import cv2

# 设置图像尺寸
size = 600
x = np.arange(1, size + 1)
Xm, Ym = np.meshgrid(x, x)

# 定义正弦波，缩放到图像尺寸，并使值适用于灰度
Z = (np.sin(Xm * 2 * np.pi / size) + 1) / 2 + (np.sin(Ym * 2 * np.pi / size) + 1) / 2

# 显示3D图像
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xm, Ym, Z, cmap=cm.viridis)
ax.set_title("3D Surface of Sine Wave")
plt.show()

# 从上方显示波形（等高线样式）
plt.imshow(Z, cmap='gray')
plt.title("Sine Wave from Above")
plt.show()

# 傅里叶变换
fftA = np.fft.fft2(Z)
fftB = np.log(np.abs(np.fft.fftshift(fftA)) + 1)
plt.imshow(fftB, cmap='gray')
plt.title("Fourier Transform of Sine Wave")
plt.show()

# 如果需要更高频率的正弦波
freq = 10
Z_high_freq = (np.sin(Xm * 2 * np.pi * freq / size) + 1) / 2 + (np.sin(Ym * 2 * np.pi * freq / size) + 1) / 2
plt.imshow(Z_high_freq, cmap='gray')
plt.title("High Frequency Sine Wave")
plt.show()

# 图像操作
# 读取图像
image_path = "picture.png"  # 替换为实际图片路径
image = Image.open(image_path)
plt.imshow(image)
plt.title("Original Image")
plt.show()

# 转为灰度
image_gray = image.convert('L')
image_array = np.array(image_gray)
plt.imshow(image_array, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# 图像的傅里叶变换
fftA = np.fft.fft2(image_array)
fftB = np.log(np.abs(np.fft.fftshift(fftA)) + 1)
fftC = fftB / np.max(fftB)
plt.imshow(fftC, cmap='gray')
plt.title("Fourier Transform of Image")
plt.show()

# 旋转图像
image_rotated = image.rotate(45)
plt.imshow(image_rotated)
plt.title("Rotated Image")
plt.show()
