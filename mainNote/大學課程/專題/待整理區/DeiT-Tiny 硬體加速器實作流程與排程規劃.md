這是一份專為 **DeiT-Tiny (Data-efficient Image Transformer)** 量身打造的硬體實作流程規劃。

DeiT-Tiny 雖然已經是輕量版，但對大學生專題來說，硬度依然很高（特別是 Softmax 和 LayerNorm 的硬體化）。這份指南將流程拆解為可執行的步驟，幫助你安排進度。

---

# DeiT-Tiny 硬體加速器實作流程與排程規劃
**目標：** 在 FPGA 上實作 DeiT-Tiny 的推論加速器，完成從 Python 模型到 Verilog 硬體的完整設計流程。

---

## 第一階段：軟體模型準備與量化 (Month 1)
**目標：** 在電腦上跑通模型，並將 32-bit 浮點數 (FP32) 轉為 8-bit 整數 (INT8)。

### 1. 模型架設 (Python/PyTorch)
*   **工具：** PyTorch, `timm` library。
*   **任務：**
    *   下載預訓練好的 DeiT-Tiny (image size 224x224)。
    *   **關鍵動作：** 寫一個簡單的 Inference Script，輸入一張測試圖，印出每一層（Layer）的輸出數值。這些數值將是你未來的「標準答案 (Golden Answer)」。
*   **產出：** `deit_tiny_float_inference.py`，以及每一層的權重檔 (Weights)。

### 2. 模型量化 (Quantization)
*   **觀念：** 硬體做浮點運算太佔面積且慢，必須轉成整數 (INT8)。
*   **任務：**
    *   採用 **Post-Training Quantization (PTQ)**。
    *   決定量化策略：Symmetric (對稱) 或 Asymmetric (非對稱)。建議先用 **Symmetric** (比較好寫 Verilog)。
    *   計算每一層的 Scaling Factor ($S$) 和 Zero Point ($Z$)。
    *   公式：$Real\_Value = S \times (Quantized\_Value - Z)$
*   **產出：**
    *   量化後的權重檔 (weights_int8.txt)。
    *   每一層的 Scaling Factors。
    *   驗證：量化後的準確率掉分不能太多 (例如 < 1-2%)。

---

## 第二階段：Golden C-Model 開發 (Month 2)
**目標：** 用 C 語言寫出「行為跟硬體一模一樣」的模擬程式。**這一步最重要，C-Model 錯，硬體一定錯。**

### 1. 基礎運算模擬
*   **任務：** 不要用 `float`，全部用 `int8_t` 和 `int32_t` (Accumulator) 來寫。
*   **實作矩陣乘法 (MatMul)：**
    *   模擬 INT8 x INT8 = INT32 的過程。
    *   實作 Re-quantization (將 32-bit 結果縮放回 8-bit 給下一層吃)。

### 2. 非線性函數近似 (Hard Part)
Transformer 的兩大魔王：**GELU** 和 **Softmax**。硬體算不動指數 ($e^x$) 和除法。
*   **GELU：** 使用 **Look-Up Table (LUT)** 查表法，或是用 ReLU 近似 (如果有重訓的話)。
*   **Softmax：**
    *   建議參考論文：*I-BERT* 或 *A Full-integer Softmax Approximation*。
    *   核心概念：用多項式 ($x^2 + ax + b$) 來近似 $e^x$，並將除法轉為乘法+位移 (Shift)。
*   **LayerNorm：** 涉及開根號 ($1/\sqrt{x}$)，同樣建議用「查表法」或「牛頓迭代法」的定點數實作。

### 3. Bit-True 驗證
*   **任務：** 把 Python 量化後的輸入丟進 C-Model，輸出的每一個 Hex 值必須跟 Python 模擬的整數輸出 **100% 一樣**。

---

## 第三階段：硬體架構設計 (Verilog RTL) (Month 3-4)
**目標：** 寫出能合成的電路。

### 1. 定義介面與總線 (Interface)
*   **資料流：** 使用 **AXI4-Stream** 介面 (Data, Valid, Ready, Last)。
*   **控制流：** 使用 **AXI-Lite** 讓 CPU (PS端) 設定參數 (如圖片張數、開始訊號)。

### 2. 核心模組設計 (Core Modules)
DeiT 的硬體通常包含三個主要 Engine：

*   **A. MatMul Engine (矩陣運算單元)：**
    *   設計一個 **Systolic Array (脈動陣列)** 或 **PE Array (4x4 或 8x8)**。
    *   負責計算 $Q \times K^T$, $A \times V$, 和 MLP 中的 Linear 層。
*   **B. Non-linear Engine (非線性單元)：**
    *   包含 Softmax, LayerNorm, GELU 的電路。
    *   裡面會有大量的 LUT (ROM) 和 DSP (乘法器)。
*   **C. Memory Controller (記憶體控制)：**
    *   負責產生讀寫 SRAM 的地址 (Address Generation Unit, AGU)。
    *   DeiT 需要處理 Patch 的切分與重組，這裡邏輯比較繁瑣。

### 3. 系統整合 (Top Integration)
*   設計一個 **Finite State Machine (FSM)** 來調度整個流程：
    *   State 0: Load Data (DMA -> SRAM)
    *   State 1: Compute Layer 1 (MatMul -> Add Bias -> GELU)
    *   State 2: Store Intermediate Result
    *   ...
    *   State N: Output Final Result

---

## 第四階段：驗證與 FPGA 實作 (Month 5)
**目標：** 跑出波形，並上板子 Demo。

### 1. Testbench 驗證 (Simulation)
*   **工具：** Vivado Simulator / ModelSim。
*   **任務：** 讀取 C-Model 產出的 Input Pattern，餵給 Verilog Top Module，比對 Output 是否與 C-Model 一致。
*   **Debug：** 這裡會花最多時間，通常是 FSM 跳錯狀態或是 Overflow。

### 2. Block Design 整合 (Vivado IP Integrator)
*   **Zynq PS (ARM):** 負責讀圖片、控制流程。
*   **AXI DMA:** 負責搬運圖片數據到你的加速器。
*   **Your IP (DeiT Accelerator):** 中間的黑盒子。

### 3. 上板實測 (On-Board Test)
*   **Python (Jupyter Notebook on PYNQ):**
    ```python
    # 偽代碼
    img = load_image("game_screenshot.jpg")
    dma.send(img)
    accelerator.start()
    result = dma.recv()
    print("Prediction:", class_labels[result])
    ```
*   **效能測量：** 計算 FPS (Frames Per Second) 和 Power (功耗)。

---

## 詳細排程表 (Gantt Chart 概念)

| 週次 | 階段 | 詳細任務 | 產出 |
| :--- | :--- | :--- | :--- |
| **W1-W2** | **軟體準備** | 閱讀 DeiT 論文、跑通 PyTorch 模型、選定 Dataset (如 CIFAR-10 resized 或 遊戲截圖) | PyTorch Inference Script |
| **W3-W4** | **量化實驗** | 實作 INT8 量化腳本、提取 Scaling Factors、驗證準確率 | Quantized Weights (.txt) |
| **W5-W7** | **C-Model** | 撰寫 Fixed-point C code、實作 Softmax/GELU LUT、**Bit-true 驗證** (最關鍵) | `model.c`, `golden_data.h` |
| **W8-W9** | **RTL: PE** | 設計 PE (MAC單元)、設計 PE Array、驗證矩陣乘法 | `pe_array.v` |
| **W10-W11**| **RTL: Logic** | 設計 Softmax/LayerNorm/GELU 電路、SRAM Controller | `softmax.v`, `layer_norm.v` |
| **W12-W13**| **RTL: Top** | 設計 FSM 控制器、整合所有模組、AXI 介面封裝 | `deit_top.v` |
| **W14-W15**| **驗證** | 跑 Simulation、Debug、修正時序 (Timing) 問題 | 波形圖 (Waveform) |
| **W16** | **FPGA** | Vivado Block Design、Bitstream 生成、上板 Demo | **Final Demo !!** |

---

## 實作小撇步 (Pro Tips for DeiT)

1.  **記憶體是瓶頸：**
    DeiT-Tiny 的參數約 5M (5MB)。大部份 FPGA 的 On-chip SRAM (BRAM) 只有 4~6MB 甚至更少。
    *   *解法：* 權重 (Weights) 可能放不下，需要放在外部 DDR (DRAM)，運算時透過 DMA 分批搬進來 (Tiling)。這會增加 FSM 的複雜度，請預留時間處理。

2.  **Softmax 可以偷懶：**
    標準 Softmax 是 $e^x / \sum e^x$。但在 Inference 時，如果你只在乎「誰是最大值」(Classification)，最後一層其實不需要做 Softmax，直接比大小就好。
    *   *注意：* 中間層 (Self-Attention) 的 Softmax **不能省**，還是要乖乖做近似。

3.  **善用 Vivado HLS (High-Level Synthesis) [選用]：**
    如果你的 Verilog 寫不完，可以考慮用 **Vivado HLS** (C++ 轉 Verilog) 來實作複雜的 Softmax/LayerNorm 模組，然後在 Verilog 中把它當 IP 呼叫。這會大幅縮短開發時間（雖然教授可能希望你手刻 Verilog）。

4.  **先求有，再求好：**
    先做一個「只有一層 Layer」或「只有 4 個 Head」的縮減版 DeiT 跑通流程，確認介面沒問題，再把層數加回去。

這份流程非常標準且紮實，如果你能照著走完，這絕對是一個 A+ 級別的畢業專題。祝你實作順利！