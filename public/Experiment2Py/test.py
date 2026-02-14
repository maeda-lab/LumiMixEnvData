from psychopy import visual, core, event
import numpy as np

# 画面の設定
win = visual.Window([1920, 1080], color="gray", units="pix")

# 刺激の初期位置
start_x = 0

# **上部刺激（コントロール）：100%-0% の輝度混合**
stim_top = visual.GratingStim(win, tex="sin", mask="gauss", size=300, pos=(start_x, 200), sf=0.015)
stim_top.tex = np.ones((256, 256)) * 1.0  # 左視点のみ（100%-0%）

# **下部刺激（調整可能）：50%-50% の輝度混合**
stim_bottom = visual.GratingStim(win, tex="sin", mask="gauss", size=300, pos=(start_x, -200), sf=0.015)
stim_bottom.tex = np.ones((256, 256)) * 0.5  # 50%-50% の輝度混合

# **速度設定**
speed_top = 5.0  # 上の刺激の速度（固定）
speed_bottom = 5.0  # 下の刺激の速度（被験者が調整）

# **実験ループ**
while True:
    keys = event.getKeys(keyList=["up", "down", "return", "escape"])
    
    for key in keys:
        if key == "up":
            speed_bottom += 0.1  # 速度を上げる
        elif key == "down":
            speed_bottom -= 0.1  # 速度を下げる
        elif key == "return":
            print(f"決定速度: {speed_bottom}")
            core.quit()
        elif key == "escape":
            core.quit()

    # **上部の刺激（コントロール刺激）**
    stim_top.pos = (stim_top.pos[0] + speed_top / 60.0, 200)
    
    # **下部の刺激（被験者が調整する刺激）**
    stim_bottom.pos = (stim_bottom.pos[0] + speed_bottom / 60.0, -200)

    # **描画と更新**
    stim_top.draw()
    stim_bottom.draw()
    win.flip()
