import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# CONFIG
# ======================
d = 0.9 * np.pi
N = 2000

# ======================
# DATA
# ======================
t = np.linspace(0.0, 1.0, N)

# (1) Nonlinear curve in your existing figure:
# k(t) = atan2( t sin d, (1-t) + t cos d )
num = t * np.sin(d)
den = (1.0 - t) + t * np.cos(d)
k = np.arctan2(num, den)

# (2) Linear dashed line with same start/end as k(t)
k0, k1 = k[0], k[-1]
k_lin = k0 + (k1 - k0) * t

# (3) Screenshot bottom formula curve: w_ph(t) (alpha ≈ 1)
# u(t) = tan(k*(t)), k*(t) = t * d  (here normalized T=1)
# w_ph(t) = u / (sin d + u (1 - cos d))
k_star = t * d
u = np.tan(k_star)

# avoid potential numerical issues near k_star = pi/2
eps = 1e-12
u = np.clip(u, -1/eps, 1/eps)

w_ph = u / (np.sin(d) + u * (1.0 - np.cos(d)))

# ======================
# PLOT
# ======================
plt.figure(figsize=(5.0, 3.6))
plt.plot(t, k, linewidth=2.5, label=r"$k(t)$")
plt.plot(t, k_lin, linestyle="--", linewidth=2.0, label=r"linear (match endpoints)")
plt.plot(t, w_ph, linestyle="-.", linewidth=2.2, label=r"$w_{\mathrm{ph}}(t)$ (Eq. 4.9)")

plt.xlabel("t", fontsize=22)
plt.ylabel("value", fontsize=22)
plt.xlim(0, 1)

plt.legend()

# Save to the SAME folder as this .py file
out_dir = Path(__file__).resolve().parent
out_png = out_dir / "k_and_wph_d_0p9pi.png"

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

print("Saved:", out_png)
