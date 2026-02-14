import numpy as np
import matplotlib.pyplot as plt

alpha = 1.0
ds = [0.3*np.pi, 0.6*np.pi, 0.9*np.pi]
colors = {0.3*np.pi: "b", 0.6*np.pi: "g", 0.9*np.pi: "r"}
labels = {0.3*np.pi: r"d=0.3$\pi$", 0.6*np.pi: r"d=0.6$\pi$", 0.9*np.pi: r"d=0.9$\pi$"}

t = np.linspace(0, 1, 4001)

def k_of_t(t, d, alpha=1.0):
    k = np.arctan2(t*np.sin(d), (1-t)*alpha + t*np.cos(d))
    k = np.unwrap(k)
    return k - k[0]

def c_of_t(t, d, alpha=1.0):
    c2 = (alpha**2)*(1-t)**2 + t**2 + 2*alpha*(1-t)*t*np.cos(d)
    return np.sqrt(np.maximum(c2, 0.0))

k_curves = {d: k_of_t(t, d, alpha) for d in ds}
c_curves = {d: c_of_t(t, d, alpha) for d in ds}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=150)

# ---------- (a) Speed curve ----------
ax = axes[0]
for d in ds:
    ax.plot(t, k_curves[d], colors[d], lw=2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 2.9)
ax.set_ylabel("k")
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.arange(0, 3.0, 0.5))
ax.set_xlabel("t")   # 添加横坐标标签
ax.tick_params(direction="out", length=3, width=0.8)

# d labels: vertically aligned, placed slightly to the right of the curve (offset in points) with white bbox
x_label_a = 0.72
for d in ds:
    y_on_curve = np.interp(x_label_a, t, k_curves[d])
    ax.annotate(
        labels[d],
        xy=(x_label_a, y_on_curve),
        xytext=(8, 0),                 # 偏移 8 points 向右，避免和线重合
        textcoords="offset points",
        ha="left", va="center",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
        zorder=5
    )

# ---------- (b) Change in contrast ----------
ax2 = axes[1]
for d in ds:
    ax2.plot(t, c_curves[d], colors[d], lw=2)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("c")
ax2.set_xticks(np.linspace(0, 1, 6))
ax2.set_yticks(np.linspace(0, 1, 6))
ax2.set_xlabel("t")  # 添加横坐标标签
ax2.tick_params(direction="out", length=3, width=0.8)

# d labels for panel b: same处理
x_label_b = 0.52
for d in ds:
    y_on_curve = np.interp(x_label_b, t, c_curves[d])
    ax2.annotate(
        labels[d],
        xy=(x_label_b, y_on_curve),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left", va="center",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
        zorder=5
    )

# captions
plt.subplots_adjust(wspace=0.35, bottom=0.30)
for ax_i, caption in zip(axes, ["(a) Speed curve", "(b) Change in contrast"]):
    bb = ax_i.get_position()
    fig.text((bb.x0 + bb.x1) / 2, 0.08, caption, ha="center", va="center", fontsize=10)

plt.show()
