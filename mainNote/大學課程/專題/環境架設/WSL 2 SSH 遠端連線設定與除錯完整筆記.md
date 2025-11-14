---
banner: 模板/banner.jpg
banner_y: "87"
---
這份 Markdown 筆記分為兩大部分：
1.  **最終正確的設定教學**：給未來的你，或任何需要設定的人，提供一份乾淨、直接、可複製貼上的操作指南。
2.  **我們的偵錯全紀錄**：詳細記錄了我們是如何一步步排除故障的，這對於學習系統管理和解決問題的思路非常有價值。

---

# WSL 2 SSH 遠端連線設定與除錯完整筆記

## 最終目標
允許從區域網路中的任何設備，透過 SSH 連線到 Windows 電腦上執行的 WSL 2 (Ubuntu/Debian) 環境。

---

## 🚀 一、最終正確的設定教學 (The How-To Guide)

這是在一個全新的環境中，從零到一設定成功的完整步驟。

### 步驟 1：以系統管理員身分開啟 PowerShell

點擊「開始」，輸入 `PowerShell`，右鍵點擊「Windows PowerShell」，選擇「以系統管理員身分執行」。**後續所有 Windows 指令都在此視窗執行。**

### 步驟 2：在 WSL 中安裝並設定 OpenSSH 伺服器

```bash
# 更新套件列表
sudo apt-get update

# 安裝 OpenSSH Server
sudo apt-get install openssh-server -y

# 建立 SSH 服務正常運作所需的目錄
sudo mkdir -p /run/sshd
sudo chmod 0755 /run/sshd

# 編輯 SSH 設定檔 (可選，但建議檢查)
sudo nano /etc/ssh/sshd_config
# 將 `#PasswordAuthentication yes` 改為 `PasswordAuthentication yes` 以啟用密碼驗證。
# 將 `#Port 22` 改為 `Port 22` 以啟用預設埠

# 啟動 SSH 服務並設定開機自啟

sudo systemctl enable ssh
sudo systemctl start ssh

# 檢查服務狀態，確保顯示綠色的 "active (running)"
sudo systemctl status ssh
```

### 步驟 3：在 Windows PowerShell 中設定網路轉發

```powershell
# 取得 WSL 2 的內部 IP 位址並存到變數中
$wsl_ip = (wsl -e hostname -I).Trim()

# 建立從 Windows 2222 連接埠到 WSL 22 連接埠的轉發規則
# listenaddress=0.0.0.0 讓區域網路中的所有設備都能存取
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=$wsl_ip

# 建立 Windows 防火牆規則，允許外部流量進入 2222 連接埠
netsh advfirewall firewall add rule name="WSL2 SSH" dir=in action=allow protocol=TCP localport=2222
```

### 步驟 4：開始連線！

1.  **取得你的 Windows 電腦在區域網路中的 IP**，在 PowerShell/CMD 中執行 `ipconfig`，找到 IPv4 位址（例如 `192.168.1.100`）。
2.  **取得你在 WSL 中的使用者名稱**，在 WSL 中執行 `whoami`（例如 `undertaker4141`）。

從任何 SSH 客戶端執行：
```bash
ssh <你的WSL使用者名稱>@<你的Windows主機IP> -p 2222

# 範例:
ssh undertaker4141@192.168.1.100 -p 2222
```

---

## 🕵️‍♂️ 二、我們的偵錯全紀錄 (The Debugging Journey)

這是一段曲折但極具學習價值的過程，記錄了我們如何從一個簡單的錯誤，層層深入，最終找到多個問題根源的完整歷程。

### 階段一：初步設定與權限問題

*   **問題**：`netsh` 指令回報 `The requested operation requires elevation (Run as administrator)`。
*   **原因**：未使用系統管理員權限執行 PowerShell。
*   **解決**：以系統管理員身分開啟 PowerShell。

*   **問題**：防火牆指令回報 `One or more essential parameters were not entered`。
*   **原因**：`netsh advfirewall` 指令缺少 `name` 和 `action` 參數，且所有參數需在同一行提供。
*   **解決**：補全指令為 `netsh advfirewall firewall add rule name="WSL2 SSH" dir=in action=allow protocol=TCP localport=2222`。

### 階段二：PowerShell 與 Linux 指令的差異

*   **問題**：在 PowerShell 中執行 `wsl -e ... | grep ...` 報錯 `The term 'grep' is not recognized`。
*   **原因**：PowerShell 不認識 `grep`, `awk` 等 Linux 工具。管道符 `|` 將輸出傳給了 PowerShell 而非 WSL 內部。
*   **解決**：改用純 PowerShell 的字串比對方法 `($output -match 'inet ([\d\.]+)').Split(' ')[1]` 或 `(wsl -e hostname -I).Trim()` 來獲取 IP。

### 階段三：核心謎團 - `Connection reset`

這是我們花費最多時間的部分，錯誤訊息始終是 `kex_exchange_identification: read: Connection reset`。

1.  **初步診斷 (`ssh -vvv`)**：
    *   **發現**：TCP 連線成功建立 (`Connection established`)，但在交換 SSH 版本資訊的第一步就被對方（WSL）重設。
    *   **結論**：問題 100% 出在 WSL 內部的 SSH 伺服器上。

2.  **第一層挖掘 (`sshd -d`)**：
    *   **工具**：使用 `sudo /usr/sbin/sshd -d` 在前景除錯模式下啟動 SSH 伺服器。
    *   **發現**：伺服器啟動時直接報錯 `Missing privilege separation directory: /run/sshd`。
    *   **解決**：手動建立目錄 `sudo mkdir /run/sshd`。

3.  **第二層挖掘 (再次 `sshd -d`)**：
    *   **問題**：即使建立了目錄，手動啟動 `sshd -d` 仍然報錯 `Bind to port 22 on 0.0.0.0 failed: Address already in use`。
    *   **發現**：`sudo service ssh stop` 指令並未完全釋放 22 連接埠，因為現代 Linux 系統使用 `ssh.socket` 進行「Socket 啟用」。
    *   **解決**：使用 `sudo systemctl stop ssh.socket` 和 `sudo systemctl stop ssh.service` 徹底停止服務並釋放連接埠。再使用 `sudo systemctl start ssh.service` 以正確的方式啟動。

4.  **第三層挖掘 (最終突破)**：
    *   **現象**：在我們修復了 WSL 內部所有問題後，`localhost` 連線依然 `Connection reset`，但**直接連線 WSL 的 IP 是成功的**！
    *   **結論**：這證明了 WSL 端的 SSH 伺服器已完全健康，問題必定出在 Windows 的 `netsh` 轉發規則本身。

5.  **真相大白 (`netsh interface portproxy show all`)**：
    *   **工具**：檢查 `netsh` 的現有規則。
    *   **發現**：規則的 `Connect to ipv4: Address` 欄位是**空的**！這是一條在早期除錯時，因 `$wsl_ip` 變數為空而建立的**損壞規則**。
    *   **最終解決**：使用 `netsh ... delete ...` 刪除損壞規則，然後用正確的 `$wsl_ip` 變數重新建立一條健康的規則。

### 核心學習點

*   **`sudo` 的重要性**：修改系統設定檔或執行系統指令，永遠不要忘記 `sudo`。
*   **`ssh -vvv`**：除錯 SSH 連線問題的第一神器，能告訴你問題發生在哪個階段。
*   **`sshd -d`**：除錯 SSH **伺服器端**問題的終極武器，能直接顯示伺服器崩潰的原因。
*   **`systemctl` vs `service`**：在現代 Linux 系統中，`systemctl` 提供了更全面的服務管理，特別是對於使用 Socket 啟動的服務。
*   **驗證的重要性**：在設定後，務必使用 `show` 或 `status` 指令（如 `netsh interface portproxy show all` 和 `sudo systemctl status ssh`）來驗證你的設定是否如預期般生效。

---
### 參考網站
[# WSL 2 Setup for SSH Remote Access  WSL 2 設定遠端 SSH 存取](https://medium.com/@wuzhenquan/windows-and-wsl-2-setup-for-ssh-remote-access-013955b2f421)