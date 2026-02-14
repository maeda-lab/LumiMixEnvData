import numpy as np
from psychopy import visual, core, event, data
import random

NUM_FRAMES = 360*4

def func(time):
    d_p_initial = 10.0
    v_P = -d_p_initial/NUM_FRAMES  # Speed parameter
    d_e = -1.0  # Eye distance
    x_p = 10.0  # Point in depth
    d_p = d_p_initial + v_P * time  # Depth of point P as a function of time

 
 

    # Calculate x (position on the screen) and v_x (velocity on the screen)
    x_screen = x_p * (-d_e) / (d_p - d_e)
    v_screen = x_p * (-d_e * v_P) / (d_p - d_e)**2
    return (x_screen, v_screen)

# PsychoPy ウィンドウの設定
win = visual.Window([1920, 1080], color="gray", units="pix")

# 固視点の設定
fixation_point = visual.Circle(win, radius=3, fillColor="red", lineColor="red", pos=(0, 0))

# 左右刺激の設定
left_stim = visual.GratingStim(win, tex="sin", mask="gauss", size=300, pos=(-200, 0), sf=0.015)
right_stim = visual.GratingStim(win, tex="sin", mask="gauss", size=300, pos=(200, 0), sf=0.015)

# # 固視点を表示
# fixation_point.draw()
# win.flip()
# core.wait(1.0)  # 1秒間の固視

left_phase = 0
right_phase = 0

t_d = np.linspace(0, NUM_FRAMES, NUM_FRAMES) # dense time points
t_s = np.linspace(0, NUM_FRAMES, int(NUM_FRAMES / 360))  # sparse time points
x_origin, _ = func(t_d)
x_sample, _ = func(t_s)
x_interp = np.interp(t_d, t_s, x_sample)

for frame in range(NUM_FRAMES):  # 5秒間（60フレーム × 5）

    # スキップキーの確認
    keys = event.getKeys(keyList=["space", "escape"])
    if "space" in keys:  # スペースキーでスキップ
        frame = 1
    if "escape" in keys:  # エスケープキーで終了
        break
        
    # 左側の刺激 (original 条件)
    left_phase = x_origin[frame] 
    left_phase %= 1  # Modulus 1 to ensure phase wraps around
    left_stim.phase = [left_phase, 0]
    left_stim.draw()

    # 右側の刺激 (sample_and_interpolate 条件)
    right_phase = -x_interp[frame] 
    right_phase %= 1  # Modulus 1 to ensure phase wraps around
    right_stim.phase = [right_phase, 0]
    right_stim.draw()
    
    # 固視点を表示
    fixation_point.draw()
    win.flip()

# 終了処理
win.close()
core.quit()
