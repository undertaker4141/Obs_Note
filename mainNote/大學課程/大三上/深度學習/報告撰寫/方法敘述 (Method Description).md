## 2. 方法敘述 (Method Description)

本章節將深入剖析本次實驗所評估之 SIRR 模型的網絡架構與核心機制。我們依據技術演進的脈絡，選取了代表不同演算法範式（Paradigm）的模型進行介紹，隨後詳細說明我們的實驗環境與訓練策略。

### 2.1. 模型架構詳述 (Detailed Model Architectures)

#### **[1] Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements (CVPR 2019)**
此模型（ERRNet）是較早期利用 CNN 解決真實世界數據匱乏問題的代表作。
*   **對齊不變損失 (Alignment-invariant Loss):** 作者觀察到真實世界的反射影像對（Reflection Pairs）往往存在些微的像素位移，直接使用像素級損失（Pixel-wise Loss）會導致訓練失敗。因此，ERRNet 引入了一種對齊不變的損失函數，使其能利用未完全對齊的真實數據進行訓練。
*   **上下文編碼模組 (Context Encoding Modules):** 為了捕捉更豐富的語義資訊，該模型在解碼器中嵌入了通道注意力（Channel Attention）與多尺度空間特徵聚合機制，強化網絡對反射特徵的判別能力。

#### **[2] Single Image Reflection Removal Based on Knowledge-Distilling Content Disentanglement (SPL 2022)**
此方法採用了知識蒸餾（Knowledge Distillation, KD）策略來解決內容解耦的難題。
*   **反射教師網絡 (Reflection Teacher Network):** 作者認為反射層相較於透射層包含較少的語義資訊，較容易學習。因此，首先訓練一個教師網絡專門提取反射層的特徵。
*   **內容解耦學生網絡 (Content Disentangling Student Network):** 學生網絡負責將輸入影像分解為透射與反射層。透過最小化學生網絡提取的反射特徵與教師網絡之間的差異（Mimicking Loss），強制學生網絡學習到正確的特徵分佈，從而實現更精確的圖層分離。

#### **[3] Revisiting Single Image Reflection Removal in the Wild (CVPR 2024)**
此模型的核心貢獻在於解決真實場景中「反射定位困難」的問題，避免虛擬反射物體被誤判為真實物體。
*   **最大反射濾波器 (Maximum Reflection Filter, MaxRF):** 基於真實背景在梯度域強度通常高於反射層的物理特性，MaxRF 通過比較影像對的梯度，能顯式地（Explicitly）提取出反射層的空間位置線索，區分出「虛擬反射」與「真實背景」。
*   **級聯式網絡架構 (Cascaded Framework):** 採用「先偵測、後移除」策略。首先由反射檢測網絡（RDNet）利用 MaxRF 的先驗估計反射置信圖（Reflection Confidence Map），接著反射移除網絡（RRNet）利用該置信圖作為注意力機制，引導網絡專注修復反射區域，保護非反射區域的細節。

#### **[4] Dual-Stream Interactive Transformers (DSIT) (NeurIPS 2024)**
DSIT 旨在解決 CNN 在捕捉全域特徵上的侷限性，以及傳統雙流網絡交互不足的問題。
*   **雙架構交互編碼器 (Dual-Architecture Interactive Encoder, DAIE):** 創新地結合了預訓練 Transformer（提供全域語義先驗）與雙流 CNN（捕捉局部細節）。透過跨架構交互（CAI）模組，將全域語義注入局部特徵流中。
*   **雙注意力交互機制 (Dual-Attention Interaction, DAI):** 包含「雙流自注意力」與「雙流交叉注意力」。前者強化各自流內的特徵表達，後者允許透射流與反射流進行顯式的資訊交換，自動抑制無關特徵，實現更精確的解耦。

#### **[5] Reversible Decoupling Network (RDNet) (CVPR 2025)**
RDNet 從訊息理論出發，解決傳統網絡因下採樣導致高頻資訊丟失的問題。
*   **多列可逆編碼器 (Multi-Column Reversible Encoder, MCRE):** 由一系列可逆單元組成，保證輸入 $x$ 可由輸出 $y$ 精確重建。這確保了在特徵提取過程中，輸入影像的所有訊息都能無損地保留至深層網絡。
*   **透射率感知提示生成器 (Transmission-rate-Aware Prompt Generator, TAPG):** 針對真實場景中反射強度受玻璃材質與光照影響的問題，TAPG 是一個輕量級網絡，能估計影像的物理參數並生成特徵提示（Prompt），動態調整主網絡權重以適應不同場景。

#### **[6] A Lightweight Deep Exclusion Unfolding Network (DExNet) (TPAMI 2025)**
DExNet 代表了「深度展開（Deep Unfolding）」技術的最新進展，兼具模型驅動的可解釋性與數據驅動的高效能。
*   **優化問題公式化:** 將 SIRR 建模為卷積稀疏編碼（CSC）問題，並引入關鍵的**廣義互斥先驗 (General Exclusion Prior)**，數學上強制透射層與反射層的特徵在空間上互斥。
*   **迭代稀疏與輔助特徵更新 (i-SAFU):** 將求解優化問題的迭代步驟展開為神經網絡層。由於網絡層共享權重，DExNet 的參數量僅約為其他 SOTA 模型的 8%，極大降低了運算成本，為一高效輕量化模型。

### 2.2. 數據集準備與劃分 (Dataset Preparation)
為了確保所有模型在相同的基準上進行比較，我們嚴格依照課程要求整理了訓練與測試數據。
*   **訓練數據集 (Training Set):** 包含總計約 14,240 對影像，來源涵蓋合成與真實場景。我們將數據整理為以下四個子目錄進行管理：
    *   `training set 1_13700`: 包含 13,700 對合成影像，利用 CEILNet 的方法生成。
    *   `training set 2_Berkeley_Real`: 包含 90 對來自 Berkeley 的真實場景影像。
    *   `training set 3_Nature`: 包含 200 對來自 Nature 數據集的真實影像。
    *   `training set 4_unaligned_train250`: 包含 250 對未對齊的 DSLR 拍攝影像。
    在訓練過程中，我們將上述數據混合，並隨機劃分為 **80% 作為訓練集 (Training Set)**，**20% 作為驗證集 (Validation Set)** 以監控收斂情況。

*   **測試數據集 (Testing Set):** 包含 7 個指定的基準數據集，用於最終評估：
    *   `Berkeley real20_420`
    *   `CEILNet_real45`
    *   `Natural Reflection Dataset (NRD)`
    *   `Nature`
    *   `SIR2` (包含 Objects, Postcard, Wild 三個子集)

### 2.3. 實作細節與實驗環境 (Implementation Details)
所有實驗均在 **Google Cloud Platform (GCP)** 的雲端運算環境中執行，硬體配置如下：
*   **GPU:** NVIDIA L4 (24GB VRAM)
*   **框架:** PyTorch

針對上述六個模型，我們均下載了官方 GitHub 提供的原始碼。為了保證結果的可復現性與公平性，我們採取了以下訓練策略：
1.  **參數設置:** 我們保持各模型官方定義的預設超參數（Default Hyperparameters），包括 Batch Size、Learning Rate 以及優化器（Optimizer，普遍使用 Adam）的設定。
2.  **數據接口:** 我們僅修改了數據讀取（Data Loader）的路徑，使其適配上述整理好的自定義數據集結構，確保所有模型都「看過」完全相同的訓練資料。
3.  **評估指標:** 訓練完成後，我們統一使用標準腳本計算 PSNR, SSIM, NCC, LMSE, LPIPS 等指標，並針對特定場景進行定性視覺比較。