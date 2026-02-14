import numpy as np
import matplotlib.pyplot as plt

# 参与者参数
participants = {
    'YAMA': {'V0': 0.992, 'A1': 0.540, 'phi1': 1.849, 'A2': -0.528, 'phi2': 1.462},
    'OMU': {'V0': 1.131, 'A1': 0.522, 'phi1': 2.528, 'A2': -0.223, 'phi2': 3.525},
    'ONO': {'V0': 1.067, 'A1': 0.632, 'phi1': 3.663, 'A2': 0.461, 'phi2': 5.123},
    'HOU': {'V0': 0.951, 'A1': 0.275, 'phi1': 3.031, 'A2': 0.920, 'phi2': 5.982},
    'LL': {'V0': 1.027, 'A1': -0.278, 'phi1': 1.849, 'A2': -0.292, 'phi2': 3.728}
}

omega = 2 * np.pi

# 时间轴 (0~4s)
t = np.linspace(0, 4, 2000)

# 颜色列表
colors = ['darkgreen', 'red', 'blue', 'orange', 'purple']

# 绘图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Velocity Functions Comparison for All Participants", fontsize=16)

# 为每个参与者创建子图
for i, (participant, params) in enumerate(participants.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    A1 = params['A1']
    phi1 = params['phi1']
    A2 = params['A2']
    phi2 = params['phi2']
    
    # 速度函数（带负号和不带负号）
    v_neg = -(A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi))
    v_pos = (A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi))
    
    ax.plot(t, v_neg, color=colors[i], linewidth=2, alpha=0.8, 
            label=f'-[A₁sin(ωt+φ₁+π) + A₂sin(2ωt+φ₂+π)]')
    ax.plot(t, v_pos, color=colors[i], linewidth=2, linestyle='--', alpha=0.8,
            label=f'A₁sin(ωt+φ₁+π) + A₂sin(2ωt+φ₂+π)')
    
    ax.set_title(f'{participant}: A₁={A1:.3f}, φ₁={phi1:.3f}, A₂={A2:.3f}, φ₂={phi2:.3f}', fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("v(t)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# 删除空的子图
axes[1, 2].remove()

# 添加总体比较图
ax_all = fig.add_subplot(2, 3, 6)
for i, (participant, params) in enumerate(participants.items()):
    A1 = params['A1']
    phi1 = params['phi1']
    A2 = params['A2']
    phi2 = params['phi2']
    
    v_neg = -(A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi))
    ax_all.plot(t, v_neg, color=colors[i], linewidth=2, label=f'{participant} (negative)')

ax_all.set_title('All Participants - Negative Functions', fontsize=10)
ax_all.set_xlabel("Time (s)")
ax_all.set_ylabel("v(t)")
ax_all.grid(alpha=0.3)
ax_all.legend(fontsize=8)
plt.tight_layout()
plt.show()
