---
banner: 模板/banner.jpg
banner_y: "87.5"
---
這份筆記整理自 [微軟官方教學文件](https://www.google.com/url?sa=E&q=https%3A%2F%2Flearn.microsoft.com%2Fzh-tw%2Fwindows%2Fwsl%2Ftutorials%2Fgui-apps)，內容涵蓋了在 Windows 上透過 WSLg 執行 Linux GUI 應用程式的核心步驟與重點。

## 什麼是 WSLg？

WSLg (Windows Subsystem for Linux GUI) 是一個開源專案，它讓您能夠在 Windows 桌面上原生執行 Linux GUI 應用程式，無需額外安裝虛擬機器或第三方 X 伺服器。應用程式會像原生 Windows 應用程式一樣顯示在工作列上，並整合到開始功能表中。

---

## 一、必要條件

在開始之前，請確保您的系統符合以下要求：

1. **Windows 版本**:
    
    - **Windows 11** (任何組建版本)
        
    - **Windows 10** (組建 19044 或更新版本)
        
2. **已安裝 WSL**:
    
    - 需要安裝 WSL 並設定一個 Linux 發行版 (例如 Ubuntu)。如果尚未安裝，可透過系統管理員權限的 PowerShell 或命令提示字元執行以下指令：
        
        codeBash
        
        ```bash
        wsl --install
        ```
        
3. **安裝虛擬 GPU (vGPU) 驅動程式**:
    
    - 為了讓 Linux GUI 應用程式能夠進行硬體加速渲染，您必須安裝對應的顯示卡驅動程式。
        
    - **Intel**: [Intel GPU 驅動程式](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.intel.com%2Fcontent%2Fwww%2Fus%2Fen%2Fdownload%2F785597%2Fintel-arc-iris-xe-graphics-windows.html)  
        : [Intel GPU 驅動程式](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
        
    - **AMD**: [AMD GPU 驅動程式](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.amd.com%2Fen%2Fsupport%2Fkb%2Frelease-notes%2Frn-rad-win-wsl-support)
        
    - **NVIDIA**: [NVIDIA GPU 驅動程式](https://www.google.com/url?sa=E&q=https%3A%2F%2Fdeveloper.nvidia.com%2Fcuda%2Fwsl)
        

---

## 二、安裝與更新

如果您已經安裝了 WSL，請確保其為最新版本以支援 WSLg。

1. **檢查更新並安裝**:  
    在 PowerShell 或命令提示字元中執行：
    
    codeBash
    
    ```bash
    wsl --update
    ```
    
    此指令會檢查並下載最新的 WSL 核心版本。
    
2. **強制重啟 WSL**:  
    更新後，建議執行以下指令來確保所有變更都已生效：
    
    codeBash
    
    ```bash
    wsl --shutdown
    ```
    
    下次啟動 Linux 發行版時，將會使用最新的核心。
    

---

## 三、執行 Linux GUI 應用程式

安裝和執行 GUI 應用程式的流程非常簡單：

1. **開啟您的 Linux 發行版** (例如從開始功能表點擊 "Ubuntu")。
    
2. **更新套件列表** (以 Ubuntu/Debian 為例)：
    
    codeBash
    
    ```bash
    sudo apt update
    ```
    
3. **使用套件管理器安裝應用程式**。
    
4. **在終端機中輸入應用程式的指令名稱**並按下 Enter。
    

應用程式的視窗將會直接出現在 Windows 桌面上，並且其圖示會顯示在 Windows 工作列。安裝後，您也可以在 Windows 的**開始功能表**中找到該應用程式的捷徑。

---

## 四、安裝範例

以下是一些常見的 GUI 應用程式安裝範例 (以 Ubuntu 為例)。

### 1. Gedit (文字編輯器)

- **說明**: 一個輕量級的圖形化文字編輯器，類似 Windows 的記事本。
    
- **安裝指令**:
    
    codeBash
    
    ```bash
    sudo apt install gedit -y
    ```
    
- **啟動指令**:
    
    codeBash
    
    ```bash
    gedit
    ```
    

### 2. GIMP (影像處理軟體)  2. GIMP （影像）

- **說明**: 功能強大的開源影像編輯軟體，常被視為 Photoshop 的替代品。
    
- **安裝指令**:
    
    codeBash
    
    ```bash
    sudo apt install gimp -y
    ```
    
- **啟動指令**:
    
    codeBash
    
    ```bash
    gimp
    ```
    

### 3. Nautilus (檔案管理器)  3. Nautilus (檔案)

- **說明**: GNOME 桌面環境的預設檔案管理器。
    
- **安裝指令**:
    
    codeBash
    
    ```bash
    sudo apt install nautilus -y
    ```
    
- **啟動指令**:
    
    codeBash
    
    ```bash
    nautilus
    ```
    

### 4. Google Chrome (網頁瀏覽器)

- **說明**: 由於 Chrome 不在預設的 apt 儲存庫中，需要手動下載並安裝。
    
- **安裝指令**:
    
    codeBash
    
    ```bash
    # 下載 .deb 安裝檔
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    
    # 使用 apt 安裝 (它會自動處理相依性)
    sudo apt install ./google-chrome-stable_current_amd64.deb -y
    ```
    
- **啟動指令**:
    
    codeBash
    
    ```bash
    google-chrome
    ```
    

### 5. Microsoft Edge (網頁瀏覽器)

- **說明**: 安裝適用於 Linux 的 Microsoft Edge 瀏覽器。
    
- **安裝指令**:
    
    codeBash
    
    ```bash
    # 設定 Microsoft 儲存庫
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge-dev.list'
    sudo rm microsoft.gpg
    
    # 更新套件列表並安裝 Edge
    sudo apt update
    sudo apt install microsoft-edge-stable -y
    ```
    
- **啟動指令**:
    
    codeBash
    
    ```bash
    microsoft-edge
    ```
    

---

## 五、疑難排解

### cannot open display 錯誤

如果您在啟動 GUI 應用程式時看到類似 Error: cannot open display: :0 的錯誤訊息，通常表示 DISPLAY 環境變數設定不正確。

1. **檢查 DISPLAY 變數**:  
    在您的 Linux 終端機中執行：
    
    codeBash
    
    ```bash
    echo $DISPLAY
    ```
    
    正確的輸出應該是 :0。
    
2. **手動設定 DISPLAY 變數**:  
    如果輸出不是 :0，您可以手動設定它：
    
    codeBash
    
    ```bash
    export DISPLAY=:0
    ```
    
    然後再試一次啟動您的 GUI 應用程式。為了讓這個設定永久生效，可以將 export DISPLAY=:0 加入到您的 ~/.bashrc 檔案中。