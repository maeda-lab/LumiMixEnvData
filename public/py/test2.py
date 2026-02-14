import numpy as np
from PIL import Image
import math
from pathlib import Path

def srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def estimate_lambda_near_seed(image_path, seed_x, seed_y,
                             dx_px=39.0, rad=260, patch_size=128,
                             n_points=40, min_dist=20,
                             fmin=1/200, fmax=1/4):
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[...,0] + 0.7152 * lin[...,1] + 0.0722 * lin[...,2]

    # Sobel x-gradient (强调竖直边缘的变化)
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    pad = np.pad(lum, 1, mode="edge")
    gx = sum(Kx[i,j] * pad[i:i+lum.shape[0], j:j+lum.shape[1]] for i in range(3) for j in range(3))
    mag = np.abs(gx)

    h, w = lum.shape
    x0, x1 = max(0, seed_x-rad), min(w, seed_x+rad)
    y0, y1 = max(0, seed_y-rad), min(h, seed_y+rad)
    roi = mag[y0:y1, x0:x1].copy()

    # 选 n_points 个强边缘点（带间隔抑制，避免都挤在一起）
    pts = []
    for _ in range(n_points * 20):
        idx = int(np.argmax(roi))
        val = float(roi.flat[idx])
        if val <= 0:
            break
        yy, xx = np.unravel_index(idx, roi.shape)
        px, py = x0 + xx, y0 + yy

        ok = True
        for qx, qy in pts:
            if (px-qx)**2 + (py-qy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            pts.append((px, py))
            if len(pts) >= n_points:
                break

        ys, ye = max(0, yy-min_dist), min(roi.shape[0], yy+min_dist)
        xs, xe = max(0, xx-min_dist), min(roi.shape[1], xx+min_dist)
        roi[ys:ye, xs:xe] = 0

    def est_lambda(patch):
        patch = patch - patch.mean()
        patch = patch * np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
        P = (np.abs(np.fft.fft2(patch))**2).sum(axis=0)  # sum over y => fx spectrum
        fx = np.fft.fftfreq(patch.shape[1], d=1.0)
        pos = fx > 0
        fxp, Pp = fx[pos], P[pos]
        band = (fxp >= fmin) & (fxp <= fmax)
        if not np.any(band):
            return None
        f_peak = fxp[band][np.argmax(Pp[band])]
        return 1.0 / f_peak

    ps = patch_size // 2
    lambdas = []
    for px, py in pts:
        if px-ps < 0 or px+ps > w or py-ps < 0 or py+ps > h:
            continue
        lam = est_lambda(gx[py-ps:py+ps, px-ps:px+ps])
        if lam is not None and np.isfinite(lam):
            lambdas.append(float(lam))

    if not lambdas:
        print("No lambda estimated (try larger rad/patch).")
        return None

    lambdas = np.array(lambdas)
    q10, q50, q90 = np.percentile(lambdas, [10, 50, 90])
    print(f"lambda(px) 10/50/90% = {q10:.2f}, {q50:.2f}, {q90:.2f}")

    d_mod = (2*math.pi*dx_px/q50) % (2*math.pi)
    print(f"Using Δx={dx_px}px, d(mod 2π) at median λ: {d_mod:.3f} rad = {d_mod/math.pi:.3f}π")

    # 返回可序列化的结果字典
    return {"lambdas": lambdas.tolist(), "q10": float(q10), "q50": float(q50), "q90": float(q90), "d_mod": float(d_mod)}

if __name__ == "__main__":
    here = Path(__file__).parent
    img_path = here / "1.png"
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    res = estimate_lambda_near_seed(str(img_path), seed_x=974, seed_y=54, dx_px=39.0)
    if res is not None:
        import json
        out = here / "estimate_lambda_result.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("Saved JSON result to", out)
