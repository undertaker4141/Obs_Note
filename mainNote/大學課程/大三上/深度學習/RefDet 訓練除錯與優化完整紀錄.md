# RefDet 訓練除錯與優化完整紀錄

此文件記錄了將 RefDet (CVPR2024) 訓練程式碼移植至 Windows 環境、整合自定義資料集以及解決一系列錯誤的完整過程。

## 1. 核心功能實作與優化 (Implementation & Optimization)

### 1.1 資料集整合 (Dataset Integration)
- **移植 [UnifiedDSRDataset](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/datasets/unified_dataset.py#108-249)**: 建立 [datasets/unified_dataset.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/datasets/unified_dataset.py)，支援自定義資料夾結構與自動計算反射層 (R = I - T)。
- **設定 [FusionDataset](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/datasets/datasets_pairs.py#343-365)**: 整合 `13700`, `Berkeley_Real`, `Nature`, `unaligned` 四個資料集，設定混合比例為 `[0.4, 0.2, 0.2, 0.2]`。
- **動態路徑**: 修改 [training.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/training.py) 以根據 `--training_data_path` 自動搜尋子資料夾，不再寫死絕對路徑。

### 1.2 Windows 環境適配 (Windows Adaptation)
- **DDP Backend**: 將分散式訓練後端從 Linux 專用的 `nccl` 改為 Windows 相容的 `gloo`。
- **單機模式支援**: 增加檢查邏輯，若 `local_rank` 為 -1 (單 GPU)，則跳過 DDP 初始化與相關操作。

### 1.3 效能優化 (Performance)
- **混合精度訓練 (AMP)**: 引入 `torch.cuda.amp.GradScaler` 與 `autocast`，減少顯存佔用並加速訓練。
- **DataLoader 設定**: 啟用 `prefetch_factor=2`, `persistent_workers=True`, `pin_memory=True` 以提升資料讀取效率。

## 2. 除錯紀錄 (Debugging Log)

以下按時間順序記錄了遇到的錯誤與解決方案：

### 2.1 引用與環境錯誤
- **`ImportError: cannot import name '_accumulate' from 'torch._utils'`**
  - **原因**: 舊版 PyTorch 代碼引用了新版已移除的內部函數。
  - **解法**: 修改 [datasets/datasets_pairs.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/datasets/datasets_pairs.py) 與 [datasets/torchdata.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/datasets/torchdata.py)，移除該引用 (未使用)。
- **`ImportError: cannot import name 'UnifiedDSRDataset'`**
  - **原因**: [training.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/training.py) 引用路徑錯誤。
  - **解法**: 將引用來源從 `datasets.image_folder` 改為正確的 `datasets.unified_dataset`。
- **`ImportError: cannot import name 'NAFNetLocal'`**
  - **原因**: 該類別在 [NAFNet_arch.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/networks/NAFNet_arch.py) 中不存在且未被使用。
  - **解法**: 從 [training.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/training.py) 中移除該引用。

### 2.2 路徑與權重錯誤
- **`FileNotFoundError` (EfficientNet weights)**
  - **原因**: [networks/efficientnet_pytorch/utils.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/networks/efficientnet_pytorch/utils.py) 寫死了作者的本地路徑。
  - **解法**: 恢復使用 `torch.utils.model_zoo.load_url` 自動下載預訓練權重。
- **`FileNotFoundError` (VGG19 weights)**
  - **原因**: [loss/losses.py](file:///d:/DL_term_project/Models/Reflection_RemoVal_CVPR2024/loss/losses.py) 寫死了作者的本地 `.pth` 路徑。
  - **解法**: 改用 `torchvision.models.vgg19(pretrained=True)` 自動載入標準權重。
- **`FileNotFoundError` (Evaluation Output)**
  - **原因**: 評估階段輸出的檔名 `name[0]` 包含了完整的 Windows 路徑 (如 `Datasets\testing set\...`)，直接拼接導致路徑錯誤。
  - **解法**: 使用 `os.path.basename(name[0])` 只取檔名。同時將 `unified_path` 預設值改為 `./`。

### 2.3 執行期錯誤 (Runtime Errors)
- **`TypeError: 'NoneType' object is not iterable`**
  - **原因**: `NAFNet` 參數 `enc_blks` 預設為 None。
  - **解法**: 在 `argparse` 中為 `enc_blks` 與 `dec_blks` 補上預設列表值。
- **`RuntimeError: Device index must not be negative`** (多次出現)
  - **原因**: 在單 GPU 模式下 `local_rank` 為 -1，但程式碼仍嘗試執行 `model.to(local_rank)` 或 `inputs.to(local_rank)`。
  - **解法**:
    1.  初始化模型時：增加判斷，若單 GPU 則使用 `cuda` device。
    2.  訓練迴圈輸入時：動態設定 `target_device` (`local_rank` 或 `device`)。
- **`AttributeError: 'RandomSampler' object has no attribute 'set_epoch'`**
  - **原因**: 單 GPU 使用 `RandomSampler`，不支援 DDP 專用的 `set_epoch`。
  - **解法**: 將 `set_epoch` 呼叫包裹在 `if args.local_rank != -1:` 判斷中。
- **`ValueError: too many values to unpack`**
  - **原因**: `UnifiedDSRDataset` 回傳字典 (`dict`)，但訓練迴圈預期回傳 Tuple (`inputs, label, name`)。
  - **解法**: 修改為從字典中依 key (`input`, `target_t`, `fn`) 取值。
- **`ValueError: Default process group has not been initialized`**
  - **原因**: 在評估與儲存模型時，無條件呼叫 `dist.get_rank()`，但在單機模式下未初始化 process group。
  - **解法**: 增加判斷 `if args.local_rank == -1 or dist.get_rank() == 0:`，並區分 `net.state_dict()` 與 `net.module.state_dict()`。

## 3. 最終執行指令 (Final Execution Command)

目前的程式碼已穩定，可使用以下指令進行訓練：

```bash
python training.py \
    --training_data_path "d:/DL_term_project/Datasets" \
    --BATCH_SIZE 6 \
    --EPOCH 200 \
    --learning_rate 0.0002 \
    --fusion_ratio 0.4 \
    --Crop_patches 320
```
