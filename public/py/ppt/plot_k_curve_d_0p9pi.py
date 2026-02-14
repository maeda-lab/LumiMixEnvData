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

# Nonlinear curve: k(t) = atan2( t sin d, (1-t) + t cos d )
num = t * np.sin(d)
den = (1.0 - t) + t * np.cos(d)
k = np.arctan2(num, den)

# Linear dashed line with same start/end as k(t)
k0 = k[0]
k1 = k[-1]
k_lin = k0 + (k1 - k0) * t

# ======================
# PLOT
# ======================
plt.figure(figsize=(5.0, 3.6))
plt.plot(t, k, linewidth=2.5)
plt.plot(t, k_lin, linestyle="--", linewidth=2.0)

plt.xlabel("t",fontsize=22)
plt.ylabel("k",fontsize=22)
plt.xlim(0, 1)
handles, labels = plt.gca().get_legend_handles_labels()
if labels:
    plt.legend()

# Save to the SAME folder as this .py file
out_dir = Path(__file__).resolve().parent
out_png = out_dir / "k_curve_d_0p9pi_with_linear_endpoint_match.png"

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

print("Saved:", out_png)
