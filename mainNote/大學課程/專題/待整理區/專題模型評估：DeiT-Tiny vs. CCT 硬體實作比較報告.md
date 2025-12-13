這是一份整理好的 **DeiT-Tiny vs. CCT 實作評估比較表**，專門設計讓你帶著去跟學長討論。

這份文件的重點在於 **「硬體資源 (Hardware Resources)」** 與 **「設計複雜度 (Design Complexity)」** 的對比，這些是學長們最能感同身受的技術語言。

---

# 專題模型評估：DeiT-Tiny vs. CCT 硬體實作比較報告

**目的：** 評估兩款 Vision Transformer 模型在 FPGA 硬體實作上的瓶頸與時程風險，以決定專題最終採用的架構。
**背景：** 實作者無硬體加速器設計經驗，需考量學習曲線與畢業時程。

---

## 1. 核心規格與資源需求對比

| 比較維度 | **DeiT-Tiny** (標準 ViT 架構) | **CCT** (Compact Convolutional Transformer) |
| :--- | :--- | :--- |
| **輸入尺寸** | **224 x 224 x 3** (RGB) | **32 x 32 x 3** (可支援到 64x64) |
| **Patch 數量** | ~196 Tokens | ~64 Tokens (取決於 Tokenizer) |
| **中間層資料量** | **龐大** (需 MB 級別暫存) | **極小** (僅需 KB 級別暫存) |
| **SRAM (BRAM) 需求** | **極高** (通常 FPGA 塞不下完整層) | **低** (可完全塞入 On-chip Memory) |
| **外部記憶體 (DRAM)** | **必須** (需頻繁讀寫 DDR) | **不需要** (或僅需存一次權重) |
| **介面複雜度** | **高** (需實作 AXI4-Full / DMA Tiling) | **低** (AXI-Stream / AXI-Lite 即可) |
| **預估實作工時** | **4 - 6 個月** | **2.5 - 3.5 個月** |

---

## 2. DeiT-Tiny (224x224) 實作分析

這屬於 **「標準規格」** 的挑戰，適合想深入研究記憶體架構的題目。

### 🔴 主要瓶頸 (Pain Points)
1.  **記憶體牆 (Memory Wall)：**
    *   因為 Feature Map 太大，無法一次放入 FPGA 的 SRAM (BRAM)。
    *   **硬體難點：** 必須設計複雜的 **Memory Controller**，將圖片切塊 (Tiling) 運算。意即：`讀取部分圖 -> 運算 -> 存回 DDR -> 再讀下一塊`。
    *   **風險：** 對新手來說，控制 DDR 的讀寫時序與 AXI Protocol 是最大的棄坑點。
2.  **模擬驗證極慢：**
    *   跑一張 224x224 的圖，Verilog Simulation (行為模擬) 可能需要數十分鐘甚至更久。Debug 效率極低，看波形圖會看到眼花。
3.  **Softmax 計算壓力：**
    *   Attention Map 為 $196 \times 196$，計算 Softmax 的矩陣運算量與資源消耗是大問題。

### ⏳ 時間預估
*   **前 2 個月：** 卡在解決如何讓 FPGA 與 DDR 高效傳輸資料。
*   **後 3 個月：** 才能真正開始調整 Transformer 的運算邏輯。

---

## 3. CCT (32x32) 實作分析

這屬於 **「輕量化 Edge AI」** 的挑戰，適合想快速落地應用並優化運算單元的題目。

### 🔵 主要瓶頸 (Pain Points)
1.  **混合架構 (Hybrid Arch)：**
    *   CCT 前端有一個 **Convolutional Tokenizer** (卷積層)。
    *   **硬體難點：** 除了 Transformer 的矩陣乘法引擎，還需要額外實作一個簡單的 **Line Buffer (行緩衝)** 來處理卷積運算 (3x3 Conv)。
2.  **特徵提取極限：**
    *   因為輸入圖小，若應用場景需要看極細微的細節 (如 1080p 畫面中的一粒米)，CCT 可能看不清楚。

### 🟢 相對優勢 (Why it's safer)
1.  **全片上執行 (All On-Chip)：**
    *   資料量小，所有運算都在 SRAM 內完成。**不需要碰 DDR 控制器**，省去 50% 的 Verilog 程式碼量。
2.  **快速迭代：**
    *   模擬一張 32x32 的圖非常快，改 Code -> 跑模擬 -> 看波形 -> 修正，這個循環 (Loop) 速度是 DeiT 的 10 倍以上。

---

## 4. 給學長的 Discussion Points (討論題綱)

帶著這些問題去問學長，展現你有做功課，也能確認實驗室資源：

1.  **關於記憶體資源：**
    > 「學長，我看 DeiT-Tiny 的中間層資料量好像塞不進我們板子 (如 PYNQ-Z2/ZCU102) 的 BRAM。實驗室有沒有現成的 **AXI DMA 或 DDR Controller 的範例 Code** 可以參考？如果沒有，我是不是選 CCT 避開 DDR 比較保險？」

2.  **關於非線性函數 (Softmax/GELU)：**
    > 「Transformer 裡面的 Softmax 和 GELU 很難算，學長建議我用 **LUT (查表法)** 做，還是我們有 **HLS (High-Level Synthesis)** 的授權可以用 C++ 寫再轉成 IP？」

3.  **關於除錯效率：**
    > 「因為我沒有實作經驗，擔心 Verilog Debug 很久。CCT 因為圖小，Simulation 比較快，學長覺得以我的狀況，是不是先做 CCT 拿來當 Base，行有餘力再做 DeiT 比較好？」

4.  **關於卷積層實作 (針對 CCT)：**
    > 「CCT 前面有 Conv 層，實驗室有沒有寫好的 **Line Buffer** 或是 **Sliding Window** 模組可以借用？還是我要自己重刻？」

---

## 5. 總結建議 (Conclusion)

*   **保守策略 (推薦)：** 選擇 **CCT**。
    *   **理由：** 避開 DDR 頻寬問題，專注於 Transformer 運算核心的設計。可以在 3 個月內做出成品，剩下的時間拿來做有趣的遊戲應用 Demo (太鼓達人/手勢控制)。
*   **進階策略：** 選擇 **DeiT-Tiny**。
    *   **理由：** 只有在 **「學長有現成的 DDR 存取模組可以給你用」** 的前提下才考慮。否則光是搞定資料搬運就會耗掉你大半個學期。

建議你可以跟學長說：**「我目前的想法是先用 CCT 把整個 Transformer 的硬體架構 (PE Array, Softmax, Control Unit) 跑通，因為它不需要處理複雜的外部記憶體。等這個架構穩定了，如果有時間，我再嘗試放大輸入尺寸去做 DeiT。」** 這樣進可攻退可守。


## 若有現成DDR存取模組的code的條件下，會節省多少時間？
這是一個非常好的假設性問題，這會大幅改變評估的天平。

如果學長能提供**現成、驗證過（Working）的 DDR 存取模組（例如封裝好的 AXI DMA Controller 或 AXI Master 讀寫模組）**，這將是巨大的助力。

直接回答你的問題：**這大概可以幫你節省 1 到 1.5 個月（約 4-6 週）的開發時間。**

這讓 DeiT-Tiny 從「自殺級難度」降低為「困難級難度」，**變成一個大三學生「有可能」做完的題目**。

以下是詳細的時間節省分析與剩餘的挑戰：

---

### 1. 節省了什麼？ (Saved Time: ~4-6 Weeks)

有了現成的 DDR Code，你直接跳過了最底層、最枯燥且最容易報錯的「基礎建設」階段：

1.  **避開 AXI Protocol 的深坑 (省 2-3 週)：**
    *   你不必自己去讀 AXI4 spec，不必處理 `AWVALID`, `WREADY`, `BRESP` 這些繁瑣的握手訊號時序。
    *   *現狀：* 學長的模組通常會提供簡單介面，例如：`Start_Read`, `Start_Address`, `Length`。你只要給訊號，資料就噴出來。

2.  **避開 Vivado IP 設定地獄 (省 1-2 週)：**
    *   設定 MIG (Memory Interface Generator) 或 Zynq PS DDR 的參數非常繁瑣，弄錯一個頻率或 Data Width，整個系統就不會動。現成的 Block Design 檔可以直接用。

3.  **避開基礎驗證 (省 1 週)：**
    *   你不需要寫 Testbench 去測試「記憶體能不能讀寫」，因為學長已經驗證過了。

---

### 2. 為什麼還是比 CCT 慢？ (Remaining Pain Points)

雖然省了 1.5 個月，但 DeiT-Tiny **依然比 CCT 慢且難**，原因在於**「控制邏輯 (Control Logic)」**與**「模擬速度」**。

即使有現成的卡車（DDR 模組），你還是要自己開車送貨（資料搬運邏輯）：

#### A. Tiling (切塊) 邏輯還是要自己寫
學長的 Code 通常是「通用型 DMA」（給我地址，我搬一條資料）。但 DeiT 的運算不能一次算全圖。
*   **DeiT 需求：** 圖片 224x224，切成 14x14 個 Patch。你需要寫一個 FSM (狀態機) 去指揮 DMA：
    *   Step 1: 叫 DMA 搬 Patch (0,0) 進來。
    *   Step 2: 等它搬完，開始算。
    *   Step 3: 算完，叫 DMA 把結果寫回去，同時搬 Patch (0,1)。
*   **難點：** 這個 **「指揮官 (Tiling Controller)」** 必須針對 DeiT 量身定做，學長的通用模組幫不了你。這部分邏輯很燒腦，容易 Deadlock。

#### B. Simulation (模擬) 依然龜速
*   **硬傷：** 這是物理限制。DeiT 資料量大，模擬軟體跑 224x224 的運算就是慢。
*   **影響：** 你改一行 Code，CCT 可能 5 分鐘驗證完，DeiT 可能要跑 1 小時。這會讓你的 Debug 週期變得很長。

#### C. Softmax 矩陣大小
*   DeiT 的 Attention 矩陣是 $196 \times 196$。
*   CCT 的 Attention 矩陣是 $64 \times 64$。
*   硬體實作 $196 \times 196$ 的 Softmax 還是比 $64 \times 64$ 難上許多（面積大、延遲高）。

---

### 3. 修正後的時程比較表

假設有現成 DDR Code 的情況下：

| 比較項目 | CCT (無 DDR) | DeiT-Tiny (無 DDR Code) | **DeiT-Tiny (有 DDR Code)** |
| :--- | :--- | :--- | :--- |
| **總預估工時** | **2.5 - 3.5 個月** | 4 - 6 個月 | **3 - 4.5 個月** |
| **難度等級** | ⭐⭐ (中等) | ⭐⭐⭐⭐⭐ (地獄) | ⭐⭐⭐ (困難) |
| **主要瓶頸** | 卷積層緩衝區 | 底層 DDR 介面 + Tiling | **Tiling 控制邏輯 + 模擬慢** |
| **成功率** | 90% | 50% | **75%** |

---

### 4. 關鍵建議：你要向學長確認什麼？

如果學長說「有 Code」，你要進一步確認 Code 的**完整度**，這決定了你能省多少時間：

*   **Level 1 (最基本)：** "只有 Xilinx AXI DMA IP 的 Block Design"。
    *   *省時：* 少。你還是要寫軟體驅動和硬體 FSM 來控制它。
*   **Level 2 (好用)：** "有一個寫好的 Verilog 模組，介面是 `read_req(addr, len)` 和 `data_out`"。
    *   *省時：* 中等。這就是上面評估的狀況，省下底層，要自己寫 Tiling。
*   **Level 3 (神隊友)：** "有一個專門做 CNN/矩陣運算的加速器架構，支援 Tiling 或 Double Buffering"。
    *   *省時：* **巨大**。如果是這種，做 DeiT 就跟做 CCT 差不多快了，直接選 DeiT 沒問題。

### 結論

1.  如果有現成 DDR Code，**DeiT-Tiny 變成一個「可以挑戰」的選項**，不再是不可觸碰的禁區。
2.  但即便如此，**CCT 依然比較快、比較穩**，因為「完全不用管 DDR」跟「用別人的 Code 管 DDR」相比，前者還是單純太多了。
3.  **最終決策建議：**
    *   如果學長的 Code 很強 (Level 2~3) 且有人可以問 -> **衝 DeiT-Tiny** (因為應用比較好做，支援大圖)。
    *   如果 Code 只是普通的 DMA IP (Level 1) 或學長很忙 -> **還是選 CCT 保平安**。
