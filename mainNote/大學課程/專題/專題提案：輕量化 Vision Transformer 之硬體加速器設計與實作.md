# 專題提案：輕量化 Vision Transformer 之硬體加速器設計與實作
**目標：** 實作 Transformer-based 的 AI 模型推論晶片 (Inference Accelerator)，包含 C-Model 驗證、Verilog RTL 設計及 FPGA 驗證。

## 1. 核心選題策略
針對大學部專題的時程與硬體資源（FPGA On-chip Memory 限制），本次選題鎖定 **「輕量化 (Lightweight)」** 與 **「混合架構 (Hybrid CNN-Transformer)」**，旨在解決 Transformer 在邊緣裝置上運算量過大與記憶體頻寬不足的問題。

---

## 2. 候選模型評估 (Model Candidates)

以下針對四款主流輕量化模型進行硬體實作難度與特性的分析：

### 推薦一：CCT (Compact Convolutional Transformer)
> **最推薦入門首選 (Best for Starter)**
> *專為小資料集與低解析度設計，硬體負擔最小。*

*   **簡介：** 針對 CIFAR-10/100 等小尺寸圖像設計的模型。引入了 Sequence Pooling 策略，移除了 Class Token，結構最精簡。
*   **優點 (Pros)：**
    *   **輸入尺寸小：** 原生支援 32x32 輸入，**SRAM 需求極低**，甚至不需要外部 DRAM 即可放入 FPGA。
    *   **結構單純：** 移除了位置編碼 (PE) 的硬體存儲需求（使用 Conv 隱式編碼）。
    *   **量化友善：** 參數量少 (約 3M)，適合做 INT8 量化實驗。
*   **缺點 (Cons)：**
    *   學術界名氣稍低於 DeiT。
    *   原生不適合處理高解析度大圖 (224x224)。
*   **硬體實作難度：** ★☆☆☆☆ (低)
    *   **瓶頸：** 幾乎沒有。適合用來建立第一個完整的 Transformer 硬體架構。

---

### 推薦二：DeiT-Tiny (Data-efficient Image Transformer)
> **標準學術參考標竿 (The Academic Standard)**
> *最標準的 ViT 縮小版，資源最多，但記憶體需求較高。*

*   **簡介：** Meta (Facebook) 提出。結構與標準 ViT 完全一致，但透過蒸餾 (Distillation) 技術在小模型上達到高準確率。
*   **優點 (Pros)：**
    *   **資源豐富：** `timm` 庫支援最完善，網路上 Pre-trained weights 最多。
    *   **架構標準：** 如果做出來，通用性最高，可宣稱支援標準 ViT 架構。
*   **缺點 (Cons)：**
    *   **輸入限制：** 通常需 224x224 輸入，**Patch 數量多 (196個)**，導致 Attention Map 計算量大 ($196 \times 196$)。
    *   **Softmax 壓力：** 在硬體上計算大矩陣的 Softmax 非常消耗資源。
*   **硬體實作難度：** ★★★☆☆ (中)
    *   **瓶頸：** 記憶體頻寬 (Memory Bandwidth) 與 Softmax/GELU 的近似電路設計。

---

### 推薦三：MobileViT (Apple)
> **工業界行動端首選 (Industry Favorite)**
> *CNN 與 Transformer 的完美結合，效能強但邏輯複雜。*

*   **簡介：** Apple 提出。前段使用 MobileNet 提取特徵，中段使用 Transformer 處理全域資訊。
*   **優點 (Pros)：**
    *   **效能極佳：** 在輕量級模型中，準確率通常優於 DeiT。
    *   **硬體優勢：** 結合 CNN，可以復用部分傳統 CNN 加速器設計。
*   **缺點 (Cons)：**
    *   **控制邏輯複雜：** 核心的 `Unfold -> Transformer -> Fold` 操作涉及複雜的記憶體位置變換 (Address Generation)，Verilog 狀態機 (FSM) 難寫。
*   **硬體實作難度：** ★★★★☆ (高)
    *   **瓶頸：** 複雜的 Data Reshape 與 Memory Controller 設計。

---

### 推薦四：LeViT
> **追求極致推論速度 (Optimized for Speed)**
> *移除位置編碼，改用 Bias，硬體運算更直接。*

*   **簡介：** 專為「推論速度」優化的架構。特點是隨著層數加深，Feature map 解析度降低（金字塔結構）。
*   **優點 (Pros)：**
    *   **硬體友善設計：** 使用 Hardswish (取代 GELU)、使用 Attention Bias (取代 Positional Embedding)，這兩點都大幅簡化了硬體算術單元設計。
*   **缺點 (Cons)：**
    *   **Multi-resolution：** 每一層的輸入尺寸都在變，硬體控制單元不夠通用的話會很難寫。
*   **硬體實作難度：** ★★★☆☆ (中高)

---

## 3. 綜合比較表 (Summary)

| 特徵 | **CCT (推薦)** | **DeiT-Tiny** | **MobileViT** | **LeViT** |
| :--- | :--- | :--- | :--- | :--- |
| **參數量** | 極小 (~3M) | 小 (~5M) | 極小 (~1-2M) | 小 (~6M) |
| **輸入尺寸** | **32x32** (極佳) | 224x224 | 256x256 | 224x224 |
| **SRAM 需求** | **低** | 高 | 中 | 中 |
| **RTL 邏輯複雜度** | **簡單** | 簡單 | **困難 (Reshape)** | 中等 |
| **Attention 機制** | 標準 | 標準 | 標準 | 優化版 (Bias) |
| **非線性函數** | GELU | GELU | Swish | **Hardswish (好做)** |
| **適合專題定位** | **硬體架構完整實作** | 標準架構相容性 | 進階記憶體控制 | 推論延遲優化 |

---

## 4. 預定實作流程 (Roadmap)

為了確保專題能順利完成（Make it work），擬定以下階段性目標：

1.  **Phase 1: Python 黃金模型建立 (Software Golden Model)**
    *   選定 **CCT** (搭配 CIFAR-10) 進行訓練。
    *   進行 **Post-Training Quantization (PTQ)**，將權重轉為 INT8 固定小數點 (Fixed-point)。
    *   驗證量化後的準確率損失 (Accuracy Drop)。

2.  **Phase 2: C-Model 模擬 (Hardware Simulation)**
    *   撰寫 C/C++ Code 模擬硬體行為（Bit-true simulation）。
    *   實作 **Softmax** 與 **LayerNorm** 的硬體近似算法 (如：I-BERT 方法或 LUT 表)。
    *   比對 C-Model 與 Python 輸出的誤差。

3.  **Phase 3: Verilog RTL 實作 (Hardware Implementation)**
    *   **PE Array 設計：** 實作 4x4 或 8x8 MAC 陣列。
    *   **Control Unit：** 設計 FSM 控制資料流 (Dataflow)。
    *   **Buffer Management：** 設計 On-chip SRAM 的 Ping-pong buffer 機制。

4.  **Phase 4: FPGA 驗證 (System Integration)**
    *   使用 Xilinx Zynq 平台。
    *   透過 **AXI4-Stream** 介面與 CPU (PS 端) 溝通。
    *   量測實際的 Latency (延遲) 與 Power (功耗)。

---

## 5. 預期遇到的挑戰與解決方案

*   **挑戰 1：Softmax 的指數運算 ($e^x$) 在硬體很難做。**
    *   *解法：* 採用 **Polynomial Approximation (多項式近似)** 或 **Look-up Table (LUT)** 加上 Log-domain trick。
*   **挑戰 2：Transformer 的矩陣乘法運算量大。**
    *   *解法：* 設計 **Systolic Array (脈動陣列)** 架構，最大化資料復用率 (Data Reuse)。
*   **挑戰 3：SRAM 空間不足。**
    *   *解法：* 優先採用 **Block-wise Processing**，將大矩陣切小塊運算，或選用 **CCT** 小圖模型避免此問題。

---

## 給教授的口頭報告重點 (Talking Points)

1.  **為什麼選 CCT？**
    *   「教授，我希望能先專注在 **『Transformer 硬體架構的完整性』** (從 Input 到 Output 全硬體執行)。如果選用標準 ViT (224x224)，大部分時間會花在處理 DDR 頻寬和記憶體調度。選 CCT (32x32) 可以讓我把重心放在 **PE Array 設計** 和 **非線性函數的優化** 上。」
2.  **關於量化 (Quantization)：**
    *   「我計畫主要做 **INT8 量化**，這對硬體面積和功耗最友善。我會參考 I-BERT 或相關論文來處理 Attention 機制中的整數運算問題。」
3.  **未來的擴充性：**
    *   「雖然目前用 CCT 做小圖，但這個 **Attention Engine (硬體 IP)** 設計好之後，未來只要加大 Buffer，是可以擴充去支援 DeiT 等大模型的。」