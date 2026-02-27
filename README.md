

---

# LumiMixEnvData

---

## 🇯🇵 日本語

### 概要

**LumiMixEnvData** は、**輝度混合および時間補間条件下における運動／速度知覚**を対象とした一連の視覚知覚実験の
**実験データおよび解析スクリプト**をまとめたリポジトリです。

本研究では、線形補間・関数混合・ガウス補間によって生成される
**非忠実（non-veridical）な連続運動信号**に対して、人間がどのように主観的速度一致を行うか、
さらにその知覚結果が **ROI（Region of Interest）レベルの局所コントラスト特性**と
どのように関係するかを検討します。

---

### リポジトリ構成

```
LumiMixEnvData/
├── public/
│   ├── AAAGaussDatav0/                          # 実験2 データ
│   ├── BrightnessFunctionMixAndPhaseData/       # 実験1 データ
│   └── py/
│       └── gauss/
│           ├── analyze_mse_by_subject_grouped_bar.py
│           ├── a-track_roi_contrast_lapvar_linear_vs_gauss_onset1s_5periods.py
│           ├── draw_roi_boxes_from_rois_csv.py
│           └── Generating_videos_from_images_linear_gauss_full_trunc3.py
```

---

### 実験1：線形輝度混合および関数混合

#### 📂 実験データ

BrightnessFunctionMixAndPhaseData
[https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData](https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData)

本実験では、以下の **2 種類の輝度生成条件** における速度知覚を比較します。

1. **線形輝度混合（Linear luminance mixing）**
   フレーム間の輝度を線形補間によって生成する条件。

2. **関数混合（Function mixing）**
   **cosine / linear / arccosine** の 3 種類の基本関数を組み合わせ、
   輝度変化の時間プロファイルを構成する条件。

これらの条件下で、輝度関数および位相関係が
主観的速度知覚に与える影響を検討します。

#### 📊 解析スクリプト

* `velocity_curve_linear_only_analysis.py`
  線形輝度混合条件における速度特性解析。

* `function_mix_analysis.py`
  関数混合条件（cosine / linear / arccosine）における
  知覚特性および位相効果の解析。

---

### 実験2：ガウス輝度混合による運動知覚

#### 📂 実験データ

AAAGaussDatav0
[https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0](https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0)

ガウス調制された輝度混合刺激を用い、
補間運動に対する主観的速度一致を評価します。

#### 📊 解析スクリプト

* `analyze_mse_by_subject_grouped_bar.py`
  参照運動と主観一致運動の **MSE（平均二乗誤差）** を被験者別に算出します。

---

### ROI コントラスト解析

#### 📊 ROI ベース解析

* `a-track_roi_contrast_lapvar_linear_vs_gauss_onset1s_5periods.py`

ROI 内の **ラプラシアン分散（Laplacian variance）** を用いて、
線形補間条件とガウス補間条件の局所コントラスト特性を比較します
（オンセット 1 秒、5 周期条件）。

---

### ROI 補助ツール

#### 🖼️ ROI 描画・切り出し

* `draw_roi_boxes_from_rois_csv.py`

CSV ファイルで定義された ROI を画像上に描画し、
対応する ROI 領域を切り出します。

---

### 刺激動画生成

#### 🎞️ 画像系列から動画生成

* `Generating_videos_from_images_linear_gauss_full_trunc3.py`

事前生成された画像系列から実験用動画を生成し、
線形補間・ガウス補間の両条件に対応します。

---

### 解析フロー概要

1. 画像系列から刺激動画を生成
2. ROI を定義・可視化
3. ROI 内コントラスト指標を算出
4. 物理的画像特性と主観的速度知覚結果を対応付けて解析

---

## 🇬🇧 English

### Overview

**LumiMixEnvData** contains experimental datasets and analysis scripts for a series of visual perception experiments investigating
**motion and velocity perception under luminance mixing, temporal interpolation, and local contrast variations**.

The project examines how **non-veridical motion signals**, generated via linear interpolation, function-based mixing, and Gaussian interpolation, are perceptually matched by human observers, and how these perceptual outcomes relate to **ROI-level image contrast statistics**.

---

### Experiment 1: Linear Luminance Mixing and Function Mixing

#### 📂 Experimental Data

BrightnessFunctionMixAndPhaseData
[https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData](https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData)

This experiment compares velocity perception under **two luminance generation conditions**:

1. **Linear luminance mixing**
   Frame-to-frame luminance is generated via linear interpolation.

2. **Function mixing**
   Luminance time profiles are constructed by combining three basic functions:
   **cosine, linear, and arccosine**.

The experiment investigates how **luminance functions and phase relationships** influence subjective velocity perception.

#### 📊 Analysis Scripts

* `velocity_curve_linear_only_analysis.py`
  Analysis of velocity characteristics under the linear luminance mixing condition.

* `function_mix_analysis.py`
  Analysis of perceptual effects and phase interactions in the function-mixing conditions
  (cosine / linear / arccosine).

---

### Experiment 2: Gaussian-Based Luminance Mixing

#### 📂 Experimental Data

AAAGaussDatav0
[https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0](https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0)

Gaussian-modulated luminance mixing is used to probe perceptual matching of interpolated motion.

#### 📊 Analysis Script

* `analyze_mse_by_subject_grouped_bar.py`
  Computes mean squared error (MSE) between reference and perceptually matched motion, grouped by subject.

---

### ROI-Based Contrast Analysis

* `a-track_roi_contrast_lapvar_linear_vs_gauss_onset1s_5periods.py`
  ROI-level contrast analysis using Laplacian variance to compare linear and Gaussian interpolation conditions.

---

### ROI Utilities and Stimulus Generation

* `draw_roi_boxes_from_rois_csv.py` – ROI visualization and cropping
* `Generating_videos_from_images_linear_gauss_full_trunc3.py` – Image-to-video stimulus synthesis

---

## 🇨🇳 中文

### 项目简介

**LumiMixEnvData** 是一个用于研究
**亮度混合与时间插值条件下的运动 / 速度知觉**的实验数据与分析代码仓库。

本项目系统研究由**线性亮度混合、函数混合以及高斯插值**产生的
**非真实连续运动信号**，以及这些信号的主观速度匹配结果
与 **ROI 局部图像对比度特性** 之间的关系。

---

### 实验一：线性亮度混合与函数混合

#### 📂 实验数据

BrightnessFunctionMixAndPhaseData
[https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData](https://github.com/jasminelong/expDataHub/tree/8e72e8e9680dc8ba884980344c53c79b2c80cd93/public/BrightnessFunctionMixAndPhaseData)

本实验比较了以下 **两种亮度生成方式** 下的速度知觉特性：

1. **线性亮度混合（Linear luminance mixing）**
   通过帧间亮度线性插值生成刺激。

2. **函数混合（Function mixing）**
   由 **cosine / linear / arccosine** 三种基本函数组合构成亮度随时间变化的函数形式。

实验重点考察亮度函数形式及其相位关系
对主观速度知觉的影响。

#### 📊 分析脚本

* `velocity_curve_linear_only_analysis.py`
  针对线性亮度混合条件的速度特性分析。

* `function_mix_analysis.py`
  针对函数混合条件（cosine / linear / arccosine）的
  知觉特性与相位效应分析。

---

### 实验二：基于高斯亮度混合的运动知觉

#### 📂 实验数据

AAAGaussDatav0
[https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0](https://github.com/maeda-lab/LumiMixEnvData/tree/master/public/AAAGaussDatav0)

使用高斯调制亮度混合刺激，研究插值运动的主观速度匹配。

#### 📊 分析脚本

* `analyze_mse_by_subject_grouped_bar.py`
  计算参考运动与主观匹配运动之间的 MSE，并按被试进行统计。

---

### ROI 局部对比度分析与工具

* `a-track_roi_contrast_lapvar_linear_vs_gauss_onset1s_5periods.py`
  ROI 局部对比度分析。

* `draw_roi_boxes_from_rois_csv.py`
  ROI 绘制与裁剪。

* `Generating_videos_from_images_linear_gauss_full_trunc3.py`
  图像序列合成实验视频。
