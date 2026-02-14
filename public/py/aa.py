import numpy as np
import matplotlib.pyplot as plt

def truncated_gauss_weights_3tap(u, N, sigma=0.6):
    """
    3-tap truncated Gaussian weights at continuous position u (in frame-index units).
    Only frames {c-1,c,c+1} are allowed (c = round(u)); others are 0.
    """
    c = int(np.round(u))
    idxs = np.array([c - 1, c, c + 1], dtype=int)
    idxs = np.clip(idxs, 0, N - 1)

    # unnormalized Gaussian
    d = (idxs.astype(np.float32) - float(u)) / float(sigma)
    w = np.exp(-0.5 * d * d)

    s = float(w.sum())
    if s > 1e-12:
        w /= s
    else:
        w[:] = 0.0
        w[1] = 1.0  # fallback

    out = np.zeros(N, dtype=np.float32)
    # 如果边界 clamp 导致 idx 重复（例如 0,0,1），把权重累加到同一帧上
    for i, wi in zip(idxs, w):
        out[i] += wi
    return out

def plot_weights(N=10, sigma=0.6, step=1.0, points_per_step=200):
    """
    Plot weight curves w_i(t) for i=0..N-1 across time u in [0, N-1].
    Here u corresponds to t/step, i.e., continuous frame index.
    """
    u_min, u_max = 0.0, float(N - 1)
    num_points = int((u_max - u_min) * points_per_step) + 1
    us = np.linspace(u_min, u_max, num_points)

    W = np.zeros((N, num_points), dtype=np.float32)
    active_count = np.zeros(num_points, dtype=int)

    for j, u in enumerate(us):
        w = truncated_gauss_weights_3tap(u, N, sigma=sigma)
        W[:, j] = w
        active_count[j] = int((w > 1e-12).sum())  # 理论上<=3（边界可能是2）

    plt.figure(figsize=(10, 4))
    for i in range(N):
        plt.plot(us * step, W[i], label=f"i={i}", linewidth=2)

    plt.xlabel("time t (sec)")
    plt.ylabel("weight w_i(t)")
    plt.title(f"3-tap Truncated Gaussian (max 3 frames), N={N}, sigma={sigma} (in frame units), step={step}s")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)

    # 在图上标注“每时刻参与帧数”
    # 你会看到大多数时候是3，靠边界时可能是2（因为 clamp）
    txt = f"active frames per t: min={active_count.min()}, max={active_count.max()} (<=3)"
    plt.text(0.01, 0.02, txt, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_weights(N=10, sigma=0.6, step=1.0, points_per_step=300)
