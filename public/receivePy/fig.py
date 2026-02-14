import matplotlib.pyplot as plt

# Times New Romanフォントの設定
plt.rcParams['font.family'] = 'Times New Roman'

# 座標設定
x_p, d_p = -3, 5  # 点Pの位置 (x_p, d_p)
d_e = -3          # 瞳位置 d_e
E = (0, d_e)      # 瞳の位置 E
P = (x_p, d_p)    # 点Pの位置

# スクリーンとの交点の計算
x_intersection = x_p * (-d_e) / (d_p - d_e)
intersection = (x_intersection, 0)

# プロットの設定
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# スクリーン (d=0) と座標軸の描画
ax.axhline(0, color='black', linewidth=1)  # スクリーン (d=0)
ax.axvline(0, color='black', linewidth=1)  # x軸

# 点のプロット
ax.scatter(*E, color='black')  # 瞳位置
ax.scatter(*P, color='black')  # 点P
ax.scatter(*intersection, color='black')  # スクリーン交点

# 直線の描画 (瞳と点Pを結ぶ線)
ax.plot([E[0], P[0]], [E[1], P[1]], color='black', linestyle='--')
ax.plot([E[0], intersection[0]], [E[1], intersection[1]], color='black', linestyle='--')

# 軸の矢印
arrow_props = dict(facecolor='black', arrowstyle='->', linewidth=1)
ax.annotate('', xy=(6, 0), xytext=(-6, 0), arrowprops=arrow_props)  # x軸矢印
ax.annotate('', xy=(0, 6), xytext=(0, -6), arrowprops=arrow_props)  # d軸矢印

# 点のラベル
ax.text(E[0] + 0.2, E[1] - 0.5, r"$E (0, d_e)$", color='black', fontsize=12)
ax.text(P[0] - 1, P[1] + 0.3, r"$P (x_p, d_p)$", color='black', fontsize=12)
ax.text(intersection[0] + 0.2, intersection[1] - 0.5, "Intersection", color='black', fontsize=12)

# 軸ラベルを追加
ax.set_xlabel("Screen Coordinate (x-axis)", fontsize=14, color='black')
ax.set_ylabel("Depth (d-axis)", fontsize=14, color='black')

# グラフの枠を削除
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# グリッドを非表示
ax.grid(False)

# 軸目盛を非表示
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

# グラフの設定
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal', adjustable='box')

# グラフの表示
plt.tight_layout()
plt.show()
