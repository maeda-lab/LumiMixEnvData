from PIL import Image
import numpy as np
import os

# ---- sRGB <-> Linear ----
def srgb_to_linear_u8(u8):
    c = u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb_u8(lin):
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1/2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)

# ---- 生成帧 ----
def generate_fade_and_composite(A_path, B_path, outA, outB, outAB,
                                n_frames=60, brightnessA=1.0, brightnessB=1.0):
    os.makedirs(outA, exist_ok=True)
    os.makedirs(outB, exist_ok=True)
    os.makedirs(outAB, exist_ok=True)

    # 读入
# 读入并裁剪（去掉下方 23px）
    A = Image.open(A_path).convert("RGBA")
    B = Image.open(B_path).convert("RGBA")

    wA, hA = A.size
    wB, hB = B.size
    A = A.crop((0, 0, wA, hA - 23))
    B = B.crop((0, 0, wB, hB - 23))

    Aa = np.array(A, np.uint8)
    Ba = np.array(B, np.uint8)

    # 转线性空间
    A_lin = srgb_to_linear_u8(Aa[..., :3]) * float(brightnessA)
    B_lin = srgb_to_linear_u8(Ba[..., :3]) * float(brightnessB)

    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0..1
        alphaA = 1.0 - t
        alphaB = t

        # A 的输出
        aA_u8 = np.full((Aa.shape[0], Aa.shape[1], 1), int(round(alphaA * 255)), np.uint8)
        A_rgb = linear_to_srgb_u8(np.clip(A_lin, 0, 1))
        A_frame = np.concatenate([A_rgb, aA_u8], axis=2)
        Image.fromarray(A_frame, "RGBA").save(os.path.join(outA, f"frame_{i+1:04d}.png"))

        # B 的输出
        aB_u8 = np.full((Ba.shape[0], Ba.shape[1], 1), int(round(alphaB * 255)), np.uint8)
        B_rgb = linear_to_srgb_u8(np.clip(B_lin, 0, 1))
        B_frame = np.concatenate([B_rgb, aB_u8], axis=2)
        Image.fromarray(B_frame, "RGBA").save(os.path.join(outB, f"frame_{i+1:04d}.png"))

        # 合成（直通道 Alpha: Out = A*αA + B*αB）
        out_lin = A_lin * alphaA + B_lin * alphaB
        out_rgb = linear_to_srgb_u8(np.clip(out_lin, 0, 1))
        out_a   = np.full((Aa.shape[0], Aa.shape[1], 1), 255, np.uint8)  # 合成结果可直接不透明展示
        out_frame = np.concatenate([out_rgb, out_a], axis=2)
        Image.fromarray(out_frame, "RGBA").save(os.path.join(outAB, f"frame_{i+1:04d}.png"))

    print(f"✅ Done. Generated {n_frames} frames into:\n{outA}\n{outB}\n{outAB}")


# ---- 使用 ----
generate_fade_and_composite(
    A_path=r"C:\Users\baian\Videos\framescut\out_0165.png",
    B_path=r"C:\Users\baian\Videos\framescut\out_0234.png",
    outA=r"C:\Users\baian\Videos\framesA1",
    outB=r"C:\Users\baian\Videos\framesB1",
    outAB=r"C:\Users\baian\Videos\framesAB1",
    n_frames=60
)
