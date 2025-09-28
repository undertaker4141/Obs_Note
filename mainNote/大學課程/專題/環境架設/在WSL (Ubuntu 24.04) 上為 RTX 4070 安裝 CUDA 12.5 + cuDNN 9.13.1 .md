---
banner: 模板/banner.jpg
banner_y: "88.5"
---
本文件旨在為使用最新硬體（如 NVIDIA RTX 40 系列 GPU）和最新 Linux 發行版（WSL 上的 Ubuntu 24.04 LTS）的開發者，提供一個清晰、穩定且官方的深度學習環境建置指南。

### **最終配置**

- **作業系統**: WSL 2 - Ubuntu 24.04 LTS
    
- **GPU**: NVIDIA GeForce RTX 4070
    
- **CUDA Toolkit**: 12.5
    
- **cuDNN**: 9.13.1
    
- **深度學習框架**: TensorFlow (最新版)
    

### **安裝前置條件**

1. 已安裝並可正常運作的 WSL 2 環境。
    
2. 在 **Windows 宿主機** 上已安裝最新的 NVIDIA Game Ready 或 Studio 驅動程式。
    
3. 熟悉基本的 Linux 命令列操作。
    

---

## 安裝步驟

### 步驟 1：清理舊環境 (可選但推薦)

如果您之前嘗試過安裝，建議先執行此步驟以確保環境乾淨。

codeBash

```bash
# 1. 移除所有可能殘留的 CUDA 和 cuDNN 相關套件
sudo apt-get --purge remove "*cuda*" "*cudnn*"
sudo apt-get autoremove

# 2. 移除舊的 NVIDIA 軟體源設定
sudo rm -f /etc/apt/sources.list.d/cuda-*.list
sudo rm -f /etc/apt/sources.list.d/cudnn-*.list
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600

# 3. 更新 apt 列表
sudo apt-get update
```

### 步驟 2：安裝 CUDA Toolkit 12.5 (使用 APT 網路源)

這是最穩定且容易維護的安裝方式。

codeBash

```bash
# 1. 安裝必要工具
sudo apt-get install -y ca-certificates curl gnupg

# 2. 添加 NVIDIA 的 GPG 金鑰
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg

# 3. 添加 CUDA 的 apt 網路儲存庫
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | sudo tee /etc/apt/sources.list.d/nvidia-cuda.list

# 4. 更新 apt 列表並安裝 CUDA 12.5
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-5
```

### 步驟 3：安裝 cuDNN 9.13.1 (使用官方 DEB 包)

這是針對 Ubuntu 24.04 最直接的安裝方式。

codeBash

```bash
# 1. 下載 cuDNN 9.13.1 的本地儲存庫設定檔
wget https://developer.download.nvidia.com/compute/cudnn/9.13.1/local_installers/cudnn-local-repo-ubuntu2404-9.13.1.1_1.0-1_amd64.deb

# 2. 使用 dpkg 安裝這個設定檔
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.13.1.1_1.0-1_amd64.deb

# 3. 將儲存庫的金鑰複製到 apt 的信任列表
sudo cp /var/cudnn-local-repo-ubuntu2404-9.13.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/

# 4. 更新 apt 列表
sudo apt-get update

# 5. 安裝與 CUDA 12.x 匹配的 cuDNN 套件
sudo apt-get -y install cudnn9-cuda-12
```

### 步驟 4：設定環境變數 (關鍵步驟)

這一步是為了讓系統能找到 CUDA 的執行檔和函式庫。

1. **編輯 ~/.bashrc 檔案**：
    
    codeBash
    
    ```bash
    nano ~/.bashrc
    ```
    
2. **在檔案的最末端，加入以下兩行**：
    
    codeBash
    
    ```bash
    export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
    
3. **儲存檔案並使其生效**：
    
    - 按下 Ctrl + X，接著按 Y，最後按 Enter。
        
    - 執行 source ~/.bashrc 或**重新開啟一個新的 WSL 終端機**。
        

### 步驟 5：最終驗證

在新的終端機視窗中執行以下指令，確認版本號是否正確。

1. **驗證 CUDA Toolkit**：
    
    codeBash
    
    ```bash
    nvcc --version
    ```
    
    預期輸出應包含 release 12.5。
    
2. **驗證 cuDNN**：
    
    codeBash
    
    ```bash
    cat /usr/include/x86_64-linux-gnu/cudnn_version.h | grep CUDNN_MAJOR -A 2
    ```
    
    預期輸出應顯示 CUDNN_MAJOR 9, CUDNN_MINOR 13, CUDNN_PATCHLEVEL 1。
    

---

## 後續步驟：安裝 TensorFlow 並驗證 GPU

1. **建立一個獨立的 Conda 環境**：
    
    codeBash
    
    ```bash
    conda create -n tf_env python=3.10 -y
    conda activate tf_env
    ```
    
2. **安裝 TensorFlow**：
    
    codeBash
    
    ```bash
    pip install tensorflow
    ```
    
3. **在 Python 中驗證 GPU**：
    
    codeBash
    
    ```bash
    python
    ```
    
    進入 Python 後，執行：
    
    codePython
    
    ```python
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    ```
    
    如果輸出 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]，則代表您的深度學習環境已成功建置並準備就緒！