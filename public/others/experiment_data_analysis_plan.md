# 2視点輝度混合手法における主観的速度等価性測定と再現性の向上 - データ分析計画

## 研究目的
遠隔操縦における映像伝送遅延の問題を解決するため、多視点映像の即時混合による等価的無遅延化手法の主観的速度等価性と再現性を評価し、改良された混合関数の効果を実証する。

## 実験設計の概要

### 実験1: 基準値測定実験
- **目的**: 各参加者の速度知覚特性の基準値を測定
- **データ**: 各参加者のデータから中位数を算出
- **使用値**: 
  - ONO: 0.583 (0.517, 0.713, 0.581, 0.583, 0.684, 1.0)
  - LL: 0.218 (0.0, 0.492, 0.471, 0.231, 0.178, 0.205)
  - HOU: 0.316 (0.163, 0.206, 0.555, 0.336, 0.295, 0.712)
  - OMU: 0.734 (0.817, 0.651, 0.551, 0.84, 0.582, 0.841)
  - YAMA: 0.615 (0.683, 0.616, 0.785, 0.583, 0.613, 0.581)

### 実験2: 速度調整実験
- **実験回数**: 6回
- **映像構成**: 上下2つの映像
  - 上：旋钮調整による速度調整映像
  - 下：輝度混合による速度映像
- **調整パラメータ**: 
  ```
  v(t) = V0 + A1·sin(ωt + φ1 + π) + A2·sin(2ωt + φ2 + π)
  ```
  - V0: 基準速度
  - A1: 第1調和成分の振幅
  - φ1: 第1調和成分の位相 (0 ... 2π)
  - A2: 第2調和成分の振幅
  - φ2: 第2調和成分の位相 (0 ... 2π)

### 実験条件
1. **線形輝度混合**: 3回実験
2. **実験1データ使用**: 3回実験

## データ分析計画

### 1. 記述統計分析

#### 1.1 基本統計量の算出
```python
# 各条件・各参加者の基本統計量
measures = ['mean', 'median', 'std', 'min', 'max', 'cv']

# 実験1データの特性分析
experiment1_stats = {
    'ONO': analyze_distribution([0.517, 0.713, 0.581, 0.583, 0.684, 1.0]),
    'LL': analyze_distribution([0.0, 0.492, 0.471, 0.231, 0.178, 0.205]),
    'HOU': analyze_distribution([0.163, 0.206, 0.555, 0.336, 0.295, 0.712]),
    'OMU': analyze_distribution([0.817, 0.651, 0.551, 0.84, 0.582, 0.841]),
    'YAMA': analyze_distribution([0.683, 0.616, 0.785, 0.583, 0.613, 0.581])
}
```

#### 1.2 パラメータ分析
- **V0**: 基準速度の傾向分析
- **A1, A2**: 振幅パラメータの調整範囲分析
- **φ1, φ2**: 位相パラメータの分布分析

### 2. 主観的速度等価性分析

#### 2.1 等価性指標の定義
```python
# 速度等価性指標
equivalence_index = |調整後速度 - 基準速度| / 基準速度

# 条件別等価性の比較
linear_equivalence = calculate_equivalence(linear_condition_data)
exp1_equivalence = calculate_equivalence(exp1_condition_data)
```

#### 2.2 条件間比較
- **線形vs実験1データ**: 対応のあるt検定
- **効果量**: Cohen's d算出
- **信頼区間**: 95%信頼区間の算出

### 3. 再現性分析

#### 3.1 試行間一貫性
```python
# 変動係数(CV)による再現性評価
cv_analysis = {
    'linear_condition': calculate_cv(linear_trials),
    'exp1_condition': calculate_cv(exp1_trials)
}

# 級内相関係数(ICC)による信頼性評価
icc_analysis = calculate_icc(repeated_measures_data)
```

#### 3.2 学習効果の検出
```python
# 試行順序効果の分析
learning_effect = analyze_trial_order_effect(trial_data)

# 時系列分析
time_series_analysis = analyze_temporal_patterns(time_series_data)
```

### 4. 個人差分析

#### 4.1 個人特性の抽出
```python
# クラスター分析による参加者分類
cluster_analysis = perform_clustering(participant_data)

# 個人別最適パラメータの抽出
individual_optimal_params = extract_optimal_parameters(individual_data)
```

#### 4.2 個人適応性の評価
```python
# 個人別改善率の算出
improvement_rate = calculate_individual_improvement(baseline, intervention)
```

### 5. 統計的検定

#### 5.1 主要な仮説検定
```python
# H1: 実験1データ使用条件の方が線形条件より速度等価性が高い
hypothesis_test_1 = paired_ttest(exp1_condition, linear_condition)

# H2: 実験1データ使用条件の方が再現性が高い
hypothesis_test_2 = paired_ttest(exp1_cv, linear_cv)

# H3: 個人差が存在する
hypothesis_test_3 = one_way_anova(participant_groups)
```

#### 5.2 多重比較補正
```python
# Bonferroni補正
corrected_p_values = apply_bonferroni_correction(p_values)
```

### 6. 高度な分析

#### 6.1 時系列解析
```python
# 速度調整パターンの周波数解析
frequency_analysis = perform_fft_analysis(adjustment_patterns)

# 位相同期性の分析
phase_synchronization = analyze_phase_coupling(reference, adjusted)
```

#### 6.2 機械学習による予測モデル
```python
# 個人特性による最適パラメータ予測
prediction_model = train_parameter_prediction_model(features, targets)

# 交差検証による性能評価
model_performance = cross_validate_model(model, data)
```

## 可視化計画

### 1. 基本可視化
- **箱ひげ図**: 条件間比較
- **散布図**: 相関関係の表示
- **時系列プロット**: 試行間変化の表示

### 2. 専門的可視化
- **極座標プロット**: 位相パラメータの分布
- **熱地図**: 参加者×条件のパフォーマンス
- **3D表面プロット**: パラメータ空間での等価性

### 3. 論文用図表
- **Figure 1**: 実験設計の概要図
- **Figure 2**: 条件間比較結果
- **Figure 3**: 個人差分析結果
- **Figure 4**: 改善効果の可視化

## 期待される結果と解釈

### 1. 予想される主要な発見
1. **実験1データ使用条件の優位性**: 線形条件より高い速度等価性
2. **個人差の存在**: 参加者間で最適パラメータが異なる
3. **再現性の改善**: 実験1データ使用により変動が減少

### 2. 統計的意義
- **効果量**: 中程度以上の効果量を期待
- **有意性**: p < 0.05での有意差を期待
- **信頼区間**: 実用的な範囲での効果を確認

### 3. 実用的示唆
1. **個人適応型システム**: 個人特性に応じた最適化の必要性
2. **リアルタイム調整**: 動的パラメータ調整の有効性
3. **システム設計**: 実用的な遠隔操縦システムへの応用

## 論文構成への貢献

### Results Section
- 記述統計と可視化結果
- 統計的検定結果
- 効果量と信頼区間

### Discussion Section
- 結果の解釈と意義
- 既存研究との比較
- 限界と今後の課題

### Conclusion Section
- 研究目的の達成状況
- 実用的な含意
- 今後の研究方向

## 実装のための推奨事項

### 1. データ管理
```python
# データ構造の標準化
data_structure = {
    'participant_id': str,
    'condition': str,  # 'linear' or 'exp1_data'
    'trial_number': int,
    'parameters': {
        'V0': float,
        'A1': float, 'phi1': float,
        'A2': float, 'phi2': float
    },
    'adjustment_time': float,
    'final_equivalence': float
}
```

### 2. 分析パイプライン
```python
# 分析フローの自動化
analysis_pipeline = [
    load_data,
    clean_data,
    calculate_descriptive_stats,
    perform_statistical_tests,
    generate_visualizations,
    export_results
]
```

### 3. 再現性の確保
```python
# 乱数シードの固定
np.random.seed(42)

# 分析設定の文書化
analysis_config = {
    'alpha_level': 0.05,
    'correction_method': 'bonferroni',
    'effect_size_threshold': 0.5
}
```

## まとめ

本分析計画により、以下の研究目的が達成されます：

1. **主観的速度等価性の定量化**: 客観的指標による評価
2. **再現性の向上効果の実証**: 統計的検定による証明
3. **個人適応の必要性の明確化**: 個人差分析による根拠
4. **実用的システムへの示唆**: 具体的な設計指針の提供

この分析計画は、あなたの研究テーマ「2視点輝度混合手法における主観的速度等価性測定と再現性の向上」の完全な実証を可能にし、学術的および実用的な価値を持つ研究成果の創出を支援します。