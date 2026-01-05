# AI 期末作業 - 專案結構總覽

> 📅 產生時間: 2025-12-29

---

## 📁 專案目錄結構

```
AI_term_project/
├── 📂 data/                          # 資料集
│   └── task1_dataset_kotae.csv       # 城市人流原始資料
│
├── 📂 notebooks/                     # 主要程式碼
│   ├── model1.ipynb                  # Model 1: Univariate Seq2Seq
│   ├── model2.ipynb                  # Model 2: DNN 時段分類
│   ├── model2_dnn_cnn_comparison.py  # Model 2: DNN vs CNN 比較
│   ├── train_multivariate_seq2seq.py # Model 1 變體: + is_weekend
│   ├── train_multivariate_timeperiod_seq2seq.py  # Model 1 變體: + 時段
│   ├── baseline_moving_average.py    # Baseline: 移動平均
│   ├── hyperparameter_experiment.py  # 超參數實驗 (28 組)
│   └── auto_label_weekend_and_train_dnn.py  # K-Means 自動標籤
│
├── 📂 models/                        # 訓練產物
│   ├── 🧠 模型權重 (.pth)
│   ├── 📊 評估結果 (.txt)
│   ├── 🖼️ 視覺化圖表 (.png)
│   └── 📂 hyperparameter_results/    # 超參數實驗結果
│
├── 📂 img/                           # 報告用圖片
└── README.md                         # 專案說明
```

---

## 📓 Notebooks & Scripts 說明

### Model 1: 時序預測 (Seq2Seq)

| 檔案                                         | 說明                                    | 輸出                                                                            |
| ------------------------------------------ | ------------------------------------- | ----------------------------------------------------------------------------- |
| `model1.ipynb`                             | **Univariate Seq2Seq** - 僅使用人數作為輸入特徵  | `seq2seq_model.pth`, `training_loss_univariate.png`                           |
| `train_multivariate_seq2seq.py`            | **Multivariate** - 加入 `is_weekend` 特徵 | `seq2seq_multivariate.pth`, `eval_log_multivariate.txt`                       |
| `train_multivariate_timeperiod_seq2seq.py` | **+ Time Period** - 加入時段 One-Hot 編碼   | `seq2seq_multivariate_timeperiod.pth`, `eval_log_multivariate_timeperiod.txt` |
| `baseline_moving_average.py`               | **Baseline** - 移動平均基準模型               | `eval_log_baseline_ma.txt`, `prediction_result_baseline_ma.png`               |
| `hyperparameter_experiment.py`             | **超參數實驗** - Hidden Size / Layers / LR | `hyperparameter_results/`                                                     |

### Model 2: 時段分類 (DNN/CNN)

| 檔案                                  | 說明                                       | 輸出                                                  |
| ------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `model2.ipynb`                        | **DNN 分類** - 早/中/晚三類                | `dnn_time_classifier.pth`, `confusion_matrix_dnn.png` |
| `model2_dnn_cnn_comparison.py`        | **DNN vs CNN 比較** - 含 3 種 DNN 架構實驗 | `*_comparison.png`, `confusion_matrix_dnn_cnn.png`    |
| `auto_label_weekend_and_train_dnn.py` | **K-Means 標籤生成** - 自動分類週末/平日   | 條碼圖視覺化                                          |

---

## 🧠 模型權重檔案 (`models/`)

| 檔案                                  | 模型                 | 大小  | 說明                                   |
| ------------------------------------- | -------------------- | ----- | -------------------------------------- |
| `seq2seq_model.pth`                   | Univariate Seq2Seq   | 44 MB | 4 層 LSTM, Hidden=256                  |
| `seq2seq_multivariate.pth`            | Multivariate Seq2Seq | 14 MB | 輸入: [人數, is_weekend]               |
| `seq2seq_multivariate_timeperiod.pth` | + Time Period        | 14 MB | 輸入: [人數, is_weekend, 時段 One-Hot] |
| `dnn_time_classifier.pth`             | DNN                  | 14 KB | 64→32→3                                |
| `cnn_time_classifier.pth`             | CNN                  | 50 KB | 1D Conv + FC                           |

---

## 📊 評估結果摘要

### Model 1: 時序預測

| 模型             | MSE        | RMSE      | MAE       | vs Baseline |
| ---------------- | ---------- | --------- | --------- | ----------- |
| Moving Average   | 2803.13    | 52.94     | 41.74     | -           |
| Univariate       | 391.88     | 19.80     | 11.97     | -86.0%      |
| **Multivariate** | **271.03** | **16.46** | **11.18** | **-90.3%**  |
| + Time Period    | 287.54     | 16.96     | 11.40     | -89.7%      |

### Model 2: 時段分類

| 模型             | 驗證準確率 |
| ---------------- | ---------- |
| DNN (32-16)      | 97.04%     |
| DNN (64-32)      | 98.52%     |
| **DNN (128-64)** | **99.26%** |
| CNN              | ~98.52%    |

### 超參數實驗 (28 組)

| 參數          | 實驗範圍              | 最佳值 |
| ------------- | --------------------- | ------ |
| Hidden Size   | 64, 128, 256          | 256    |
| Num Layers    | 1, 2, 4               | 4      |
| Learning Rate | 0.0001, 0.0005, 0.001 | 0.0005 |

---

## 🖼️ 視覺化圖表清單

### Model 1 相關

- `training_loss_univariate.png` - 訓練/驗證 Loss 曲線
- `prediction_result_baseline_ma.png` - Baseline 預測結果
- `prediction_result_multivariate.png` - Multivariate 預測結果
- `prediction_result_multivariate_timeperiod.png` - + Time Period 預測結果
- `hyperparameter_results/hyperparameter_comparison.png` - 28 組超參數對比

### Model 2 相關

- `confusion_matrix_dnn.png` - DNN 混淆矩陣
- `confusion_matrix_dnn_cnn.png` - DNN & CNN 混淆矩陣對比
- `dnn_cnn_training_comparison.png` - 訓練過程比較 (2x2)
- `dnn_cnn_validation_comparison.png` - 驗證 Accuracy/Loss 對比
- `dnn_architecture_comparison.png` - 不同 DNN 架構比較

---

## 📝 報告撰寫參考

### 建議章節結構

```
1. 緒論
   - 問題定義、資料來源

2. 資料分析
   - 城市選擇、前三大熱點網格

3. Model 1: 時序預測
   3.1 模型架構 (Seq2Seq LSTM)
   3.2 Baseline 比較 (Moving Average)
   3.3 三版本對比 (Univariate / Multivariate / +TimePeriod)
   3.4 超參數實驗 (28 組)

4. Model 2: 時段分類
   4.1 模型架構 (DNN vs CNN)
   4.2 混淆矩陣分析
   4.3 不同 DNN 架構對比

5. 討論與結論
   - 為什麼 Seq2Seq >> Moving Average
   - 額外特徵效果分析
   - 最佳模型推薦
```

### 重要數據引用

> Seq2Seq 模型相較於 Moving Average Baseline，MSE 降低 **86~90%**

> DNN (128-64) 達到 **99.26%** 驗證準確率

> 最佳超參數組合: Hidden=256, Layers=4, LR=0.0005
