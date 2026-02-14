from psychopy import visual, core, event, data
import random

# 実験パラメータ
num_trials = 10  # 試行数（各条件の繰り返し数）
# linear_speeds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4] # 条件3: 極定速から高速まで

linear_speeds = [0.02, 0.05, 0.1] # 条件3: 極定速から高速まで

# 非線形速度関数（速度が二乗に反比例する例）
def nonlinear_speed(x):
    return 1 / (x**2)

# PsychoPy ウィンドウの設定
win = visual.Window([1920, 1080], color="gray", units="pix")

# 固視点の設定
fixation_point = visual.Circle(win, radius=3, fillColor="red", lineColor="red", pos=(0, 0))

# 標準刺激と対象刺激の設定
left_stim = visual.GratingStim(win, tex="sin", mask="gauss", size=200, pos=(-250, 0),sf=.015)
right_stim = visual.GratingStim(win, tex="sin", mask="gauss", size=200, pos=(250, 0),sf=.015)

# 実験デザインの設定
conditions = []
for speed in linear_speeds:
    conditions.append({"standard": speed, "comparison_type": "linear"})
    conditions.append({"standard": speed, "comparison_type": "nonlinear"})
trials = data.TrialHandler(conditions, nReps=num_trials, method="random")

# 結果を記録するリスト
results = []

# 実験ループ
for trial in trials:
    # 固視点を表示
    fixation_point.draw()
    win.flip()
    core.wait(1.0)  # 1秒間の固視

    # 条件に基づき刺激を表示
    standard_speed = trial["standard"]
    comparison_type = trial["comparison_type"]

    left_stim.pos = [-200, 0] # 標準刺激（等速条件）
    right_stim.pos = [200, 0] # 対象刺激（比較条件）
    for frame in range(60):  # 1秒間（60フレーム）
    
        # 標準刺激（等速条件）
        left_stim.phase = [(left_stim.phase[0] + standard_speed) % 1, left_stim.phase[1]]
        left_stim.draw()
    
        # 対象刺激
        if comparison_type == "linear":
            right_stim.phase = [(right_stim.phase[0] + standard_speed) % 1, right_stim.phase[1]]
        elif comparison_type == "nonlinear":
            right_stim.phase = [(right_stim.phase[0] + nonlinear_speed(frame + 1)) % 1, right_stim.phase[1]]
        right_stim.draw()
        fixation_point.draw()
        win.flip()

    # 応答取得
    win.flip()  # 刺激を消去
    keys = event.waitKeys(keyList=["left", "right", "escape"])

    if "escape" in keys:
        
        # 結果を保存
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_results.csv", index=False)
        core.quit()

    # 結果を保存
    response = "left" if "left" in keys else "right"
    results.append({"standard_speed": standard_speed, "comparison_type": comparison_type, "response": response})

# 結果を保存
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)

# 終了処理
win.close()
core.quit()
