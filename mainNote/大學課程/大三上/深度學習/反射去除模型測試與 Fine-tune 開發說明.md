# 反射去除模型測試與 Fine-tune 開發說明

## 概述

本文件記錄深度學習期末專案的模型測試與 fine-tune 功能開發過程，包含兩個反射去除模型的完整評估流程。

---

## 評估指標

| 指標 | 說明 |
|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio |
| **SSIM** | Structural Similarity Index |
| **NCC** | Normalized Cross-Correlation |
| **LMSE** | Local Mean Squared Error |
| **LPIPS** | Learned Perceptual Image Patch Similarity |

---

## Table 1: 資料集評估

| Dataset | 數量 | 有 GT |
|---------|------|-------|
| CEILNet Real | 45 | ❌ |
| Berkeley Real | 20 | ✅ |
| SIR² - Objects | 200 | ✅ |
| SIR² - Postcard | 199 | ✅ |
| SIR² - Wild | 55 | ✅ |
| SIR² - All | 454 | ✅ |
| Nature | 20 | ✅ |
| NRD | 136 | ✅ |

---

## Table 2: 模型效率比較 (SIR² only)

| Metric | 說明 |
|--------|------|
| # of Parameters (M) | 模型參數量（百萬）|
| FLOPs (G) | 浮點運算量（十億）|
| Run time (s) | 單張圖片推理時間 |

---

## Model 1: Reflection_RemoVal_CVPR2024

### 路徑
```
/home/team06/DL_term_project/Models/Reflection_RemoVal_CVPR2024/
```

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| `utils/UTILS.py` | 新增 NCC, LMSE, LPIPS 計算 |
| `datasets/unified_dataset.py` | 新增 `NoGTDataset`、大圖自動降採樣 |
| `testing.py` | 整合所有指標、Table 2 效率計算 |
| `efficiency_only.py` | 獨立效率計算腳本 |
| `finetune.sh` | Fine-tune 腳本 |

### 執行測試
```bash
cd /home/team06/DL_term_project/Models/Reflection_RemoVal_CVPR2024
uv run python testing.py \
  --pre_model ./SIRR/Net_epoch_199__iters_474000.pth \
  --pre_model1 ./SIRR/Net_Det_epoch_199__iters_474000.pth \
  --load_pre_model True
```

### Fine-tune (產生不同結果)
```bash
# 使用不同 seed 產生不同結果
./finetune.sh 21 1    # 組別1：seed=21, 多訓練1個epoch
./finetune.sh 22 2    # 組別2：seed=22, 多訓練2個epoch
```

### 輸出位置
```
./SIRR__test_results/
├── summary.txt          # 包含 Table 2
├── efficiency.json
└── {dataset}-img/
```

---

## Model 2: RDNet (Model5)

### 路徑
```
/home/team06/DL_term_project/Models/Model5/RDNet/
```

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| `util/index.py` | 新增 LPIPS 計算 |
| `testing_all.py` | 完整測試腳本 + Table 2 + 命令列參數 |
| `finetune.sh` | 快速 Fine-tune 腳本 |

### 執行測試
```bash
cd /home/team06/DL_term_project/Models/Model5/RDNet

# 使用預設 checkpoint
uv run testing_all.py

# 使用自訂 checkpoint
uv run testing_all.py --checkpoint path/to/model.pt
```

### Fine-tune (產生不同結果)
```bash
# 快速 fine-tune：只用部分訓練資料
./finetune.sh 21 100   # 組別1：seed=21, 用100張圖
./finetune.sh 22 50    # 組別2：seed=22, 用50張圖

# 訓練完成後測試
uv run testing_all.py --checkpoint checkpoints/ytmt_ucs_sirs_finetuned_seed21/ytmt_ucs_sirs_finetuned_seed21_latest.pt
```

### 輸出位置
```
./test_results/
├── summary.txt
├── all_results.json
├── efficiency.json
├── {dataset}_metrics.json
└── {dataset}-img/
```

---

## 資料集路徑

```
/home/team06/DL_term_project/Datasets/testing set/
├── CEILNet_real45/
├── Berkeley real20_420/
├── SIR2/
│   ├── SolidObjectDataset/
│   ├── PostcardDataset/
│   └── WildSceneDataset/
├── Nature/
└── Natural Reflection Dataset(NRD)/
    ├── NCCU_I/
    ├── NCCU_T/
    └── NCCU_R/
```

---

## 已解決問題

| 問題 | 解決方案 |
|------|----------|
| CEILNet 無 GT | `NoGTDataset` 類別 |
| NRD LMSE 計算慢 | 向量化計算 + 大圖降採樣 |
| NRD GPU OOM | 自動降採樣到 1024px |
| JSON 序列化錯誤 | `convert_to_native()` 轉換 |
| 不同組別相同結果 | Fine-tune 腳本 (不同 seed) |
| RDNet 訓練太慢 | 限制訓練樣本數量 |

---

## 依賴安裝

```bash
uv add lpips thop pytorch-msssim
```

---

## 測試結果範例

### CVPR2024 Model (Fine-tuned seed21)
```
nature20: PSNR=25.11, SSIM=0.8351
real20: PSNR=21.64, SSIM=0.7978
SIR-all: PSNR=23.44, SSIM=0.8848
NRD: PSNR=20.58, SSIM=0.7221

Table 2:
  Parameters (M): 27.99
  FLOPs (G): 15.68
  Run time (s): 0.0801
```

### RDNet Model
```
real20: PSNR=20.72, SSIM=0.7602
SIR-all: PSNR=24.59, SSIM=0.8996
NRD: PSNR=21.78, SSIM=0.7681

Table 2:
  Parameters (M): 266.43
  FLOPs (G): 175.21
  Run time (s): 0.3108
```
