---
banner: 模板/banner.jpg
banner_y: "87"
---
本筆記是紀錄如何在 WSL (Ubuntu) 環境中進行 `Fcitx5` 繁體中文輸入法（新酷音注音）的安裝與設定，並將系統字型最佳化為 Windows 使用者習慣的「微軟正黑體」。

所有步驟均經過我親自驗證有效，可有效解決在 WSLg 環境下常見的 `Wayland` 衝突和字型優先權問題。


## 目錄
1.  [[#步驟一：安裝 Fcitx5 輸入法框架]]
2.  [[#步驟二：設定環境變數]]
3.  [[#步驟三：核心修正：禁用 Wayland 模組以避免衝突]]
4.  [[#步驟四：啟用與設定輸入法]]
5.  [[#步驟五：最佳化中文字型為微軟正黑體]]
6.  [[#步驟六：設定 Fcitx5 開機自動啟動]]

---


### 步驟一：安裝 Fcitx5 輸入法框架

首先，更新軟體包清單，並安裝 `Fcitx5` 的核心程式、中文輸入法引擎（新酷音注音）以及其他必要的相依套件。

```bash
sudo apt update
sudo apt install fcitx5 fcitx5-chewing fcitx5-frontend-gtk3 fcitx5-frontend-gtk2 fcitx5-frontend-qt5 fcitx5-configtool -y
```

---

### 步驟二：設定環境變數

編輯使用者家目錄下的 `.bashrc` 設定檔，讓所有圖形介面（GUI）應用程式都能正確地呼叫 Fcitx5 輸入法服務。

```bash
# 為避免重複，先刪除可能存在的舊設定
sed -i '/GTK_IM_MODULE/d' ~/.bashrc
sed -i '/QT_IM_MODULE/d' ~/.bashrc
sed -i '/XMODIFIERS/d' ~/.bashrc

# 寫入新的、正確的標準設定
echo 'export GTK_IM_MODULE=fcitx' | tee -a ~/.bashrc
echo 'export QT_IM_MODULE=fcitx' | tee -a ~/.bashrc
echo 'export XMODIFIERS=@im=fcitx' | tee -a ~/.bashrc
```

---

### 步驟三：核心修正：禁用 Wayland 模組以避免衝突

這是解決 Fcitx5 在 WSLg 環境下啟動後立即崩潰的關鍵步驟。我們透過**重新命名設定檔**的方式，讓 Fcitx5 找不到 Wayland 相關的模組，進而強制它退回至穩定可靠的 X11 模式 (XWayland) 運作。

```bash
sudo mv /usr/share/fcitx5/addon/wayland.conf /usr/share/fcitx5/addon/wayland.conf.bak
sudo mv /usr/share/fcitx5/addon/waylandim.conf /usr/share/fcitx5/addon/waylandim.conf.bak
```

---

### 步驟四：啟用與設定輸入法

完成以上設定後，**必須徹底重啟 WSL** 才能確保所有變更都正確生效。

1.  **在 Windows 的 PowerShell 或 CMD 視窗中**，執行以下指令：
    ```powershell
    wsl --shutdown
    ```

2.  重新開啟 WSL (Ubuntu) 的終端機。

3.  在新的終端機中，啟動 Fcitx5 服務並讓它在背景執行：
    ```bash
    fcitx5 &
    ```

4.  開啟圖形化的設定工具：
    ```bash
    fcitx5-configtool
    ```

5.  在「Fcitx 組態」視窗中，加入您需要的中文輸入法：
    *   取消勾選左下角的 **「僅顯示目前語言」**。
    *   在右側的搜尋框輸入 `Chewing`，找到 **酷音 (Chewing)**，選取後點擊向左的 `>` 箭頭，將它加入左邊的「目前的輸入法」清單。
    *   完成後直接關閉視窗即可自動儲存。

---

### 步驟五：最佳化中文字型為「微軟正黑體」

此步驟將解決 Linux 預設中文字型不美觀的問題，直接取用 Windows 內建的「微軟正黑體」並提升其顯示優先權。

1.  **建立指向 Windows 字型資料夾的符號連結（捷徑）：**
    ```bash
    sudo ln -s /mnt/c/Windows/Fonts /usr/share/fonts/windows
    ```

2.  **建立 Fontconfig 設定檔，強制系統優先使用微軟正黑體：**
    ```bash
    # 建立使用者字型設定的目錄
    mkdir -p ~/.config/fontconfig/conf.d/
		```


```bash
	# 寫入優先權設定檔
    cat <<EOF > ~/.config/fontconfig/conf.d/01-prefer-msjh.conf
    <?xml version="1.0"?>
    <!DOCTYPE fontconfig SYSTEM "fonts.dtd">
    <fontconfig>
        <alias>
            <family>sans-serif</family>
            <prefer>
                <family>Microsoft JhengHei UI</family>
                <family>Microsoft JhengHei</family>
            </prefer>
        </alias>
        <alias>
            <family>serif</family>
            <prefer>
                <family>Microsoft JhengHei</family>
            </prefer>
        </alias>
        <alias>
            <family>monospace</family>
            <prefer>
                <family>Microsoft JhengHei Mono</family>
                <family>Consolas</family>
            </prefer>
        </alias>
    </fontconfig>
    EOF
```

3.  **強制更新系統的字型快取，讓新設定生效：**
    ```bash
    sudo fc-cache -f -v
    ```

4.  完成後，**重新啟動所有 Linux 的 GUI 應用程式**，字型就會改變。

---

### 步驟六：設定 Fcitx5 開機自動啟動

為了方便起見，將以下指令加入 `.bashrc`，這樣每次開啟終端機時，系統就會自動檢查並在需要時啟動 Fcitx5，無需手動執行。

```bash
echo 'pidof fcitx5 >/dev/null || fcitx5 &' | tee -a ~/.bashrc
```

**至此， WSL 繁體中文環境已設定完成！**