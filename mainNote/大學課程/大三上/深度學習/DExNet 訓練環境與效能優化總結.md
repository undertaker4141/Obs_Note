我們成功解決了依賴缺失、Windows 相容性問題，並大幅優化了訓練速度。以下是本次對話的重點摘要，供您紀錄：

## 1. 環境修復與相容性 (Environment & Compatibility)
依賴補全：修復了 ModuleNotFoundError，手動安裝並寫入  pyproject.toml  的套件包括：
- kornia
- tensorboardX
- visdom
- PyWavelets (提供 pywt)
- dominate
- Windows 多進程修復：在 train_sirs.py 主要執行區塊加上 if __name__ == '__main__':，解決了 Windows 下 DataLoader 遞迴生成進程導致的 RuntimeError。

## 2. 資料集整合 (Dataset Integration)
- 統一介面：實作 UnifiedDSRDataset 類別，統一處理不同資料夾結構（如 training set 2, 3, 4）。
- 動態計算：支援 compute_r=True，能針對缺少反射層的資料集自動計算 R = Input - Transmission。
- 混合訓練：使用 FusionDataset 將 4 個資料集按 [0.4, 0.2, 0.2, 0.2] 比例混合訓練。

## 3. 效能優化 (Performance Optimization)
針對 RTX 4070 顯存不足 (VRAM Exhaustion) 與 Windows CPU 瓶頸，進行了三層優化：
- **Layer 1：混合精度 (AMP)**
	- 修改 DExNet_model_sirs.py，引入 torch.amp.GradScaler 與 autocast。
	- 效果：大幅降低顯存佔用，啟用 Tensor Core 加速。
- **Layer 2：資料載入 (Data Loading)**
	- 在 DataLoader 加入 prefetch_factor=2 與 persistent_workers=True。
	- 效果：在 Windows 開啟 nThreads=4 時，強迫 CPU 預取資料，減少 GPU 等待時間 (0.8s -> 0.2~0.4s)。
- **Layer 3：激進顯存管理 (VRAM Management)**
	- 凍結 VGG：將 Loss Network (VGG19) 參數設為 requires_grad=False，避免建立不必要的計算圖。
	- 凍結 VGG：將 Loss Network (VGG19) 參數設為 requires_grad=False，避免建立不必要的計算圖。

## 4. 最終訓練指令 (Final Command)
請使用此優化後的指令進行訓練 (預計時間：~1 天內完成)：
```bash
uv run --no-sync python train_sirs.py \
  --name test_run_fast \
  --model DExNet_model_sirs \
  --inet DExNet \
  --base_dir "d:\DL_term_project\Datasets" \
  --nThreads 4 \
  --loadSize 224 \
  --batchSize 1 \
  --display_id 0 \
  --nEpochs 20
```


## ### 環境全能修復指令 (Rescue Command)
如果未來環境又亂掉，請執行
```bash
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 && uv pip install opencv-python kornia tensorboardX visdom PyWavelets dominate
```

重新安裝 DExNet 所需要的所有套件

