此文件整理了將 **RefDet (Reflection Removal CVPR2024)** 模型移植至 Windows 環境並整合自定義資料集的所有修改操作與執行指南。

## 1. 修改摘要 (Modification Summary)

本次優化主要參考 DExNet 的實作，針對以下三點進行改進：

### 1.1 資料集基礎架構 (Dataset Infrastructure)

- **新增檔案**: 
    
    datasets/unified_dataset.py
- **核心類別**: 
    
    UnifiedDSRDataset
- **功能**:
    - **統一介面**: 可讀取不同資料夾結構的資料集 (`13700`, `Berkeley_Real`, `Nature`, `unaligned`)。
    - **自動計算反射層**: 若資料集中缺少 Reflection Layer (例如真實拍攝的資料)，設為 `compute_r=True` 可自動透過 $R = I - T$ 計算。
    - **同步影像增強**: 確保 Input, Transmission, Reflection 三者在 Resize, Crop, Rotate 時保持空間對齊。
    - **資料融合 (Fusion)**: 使用 
        
        FusionDataset 將四個資料集依比例 `[0.4, 0.2, 0.2, 0.2]` 混合訓練。

### 1.2 Windows 相容性修正 (Windows Compatibility)

- **DDP 後端**: Windows 不支援 `nccl`，已修改為自動偵測環境，若為 Windows 則切換至 `gloo` 後端。
- **多進程保護**: 將 `dist.init_process_group` 初始化移至 `if __name__ == '__main__':` 區塊內，避免 Windows 的 `spawn` 啟動方式導致 RuntimeError。
- **Worker 數量**: 在 Windows 環境下自動限制 
    
    DataLoader 的 `num_workers` (如設為 4 或 8)，避免過多執行緒導致 CPU 瓶頸或錯誤。

### 1.3 訓練效能優化 (Performance Optimization)

- **混合精度訓練 (AMP)**:
    - 引入 `torch.cuda.amp.GradScaler` 與 `autocast`。
    - 大幅降低 VRAM 使用量，並在 RTX 系列顯卡上加速訓練。
- **DataLoader 優化**:
    - 啟用 `prefetch_factor=2`：預先載入兩批資料，減少 GPU 等待時間。
    - 啟用 `persistent_workers=True`：避免每個 Epoch 重新建立 Worker 的開銷。
    - 啟用 `pin_memory=True`：加速 CPU 到 GPU 的記憶體傳輸。

---

## 2. 執行訓練 (Execution)

### 2.1 啟動指令 (Command)

請在專案根目錄下執行以下指令。請務必將 `--training_data_path` 替換為您實際的資料集根目錄路徑。

python training.py 
```bash
python training.py \
    --training_data_path "d:/DL_term_project/Datasets" \
    --BATCH_SIZE 6 \
    --EPOCH 200 \
    --learning_rate 0.0002 \
    --fusion_ratio 0.4 \
    --Crop_patches 320
```


### 2.2 關鍵參數說明 (Parameters)

|參數|預設值|說明|
|---|---|---|
|`--training_data_path`|(需指定)|資料集根目錄，程式會自動搜尋其下的 `13700`, `Berkeley_Real` 等子資料夾。|
|`--BATCH_SIZE`|`6`|批次大小。若開啟 AMP 後 VRAM 仍充足，可嘗試調大至 `8` 或 `10`。|
|`--Crop_patches`|`320`|訓練時隨機裁切的圖片大小。|
|`--fusion_ratio`|`0.7` (舊)|在此版本代碼中，我們硬編碼了融合邏輯，主要用於控制不同資料集的混合比例。|
|`--local_rank`|`-1`|DDP 參數。單卡訓練時無需理會，程式會自動處理。|

---

## 3. 程式碼修改細節 (Code Changes)

### datasets/unified_dataset.py

移植自 DExNet，包含 UnifiedDSRDataset 類別，負責處理路徑與影像讀取。

### training.py

主要修改點包含：

1. **Import**: 引入 UnifiedDSRDataset, FusionDataset, `GradScaler`。
2. **DDP Init**: 移動到 `main` 區塊並加入 Windows 判斷。
```python
if torch.cuda.is_available() and args.local_rank != -1:
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    dist.init_process_group(backend=backend)

```
3. **Dataset Loading**: 使用新的設定檔清單 (`datasets_config`) 取代舊的寫死路徑。
4. **Training Loop**:
```python
# AMP 前向傳播
with torch.cuda.amp.autocast():
    train_output = net(inputs, sparse_out.detach())
    # ... 計算 loss ...
# AMP 反向傳播
scaler.scale(g_loss).backward()
scaler.step(optimizerG)
scaler.update()
```