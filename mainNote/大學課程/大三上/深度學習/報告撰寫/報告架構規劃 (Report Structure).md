恭喜你們完成了最耗時的模型訓練階段！由於這是一份期末專題報告（Term Project），且佔總成績的 **35%**，加上有明確的格式與內容要求（至少 10 頁），報告的撰寫策略應該要**「結構嚴謹、論述有據、圖表精美」**。

以下是根據你們提供的 PDF 講義內容與參考文獻，為你們規劃的**書面報告架構**與**分工建議**：

---

### 一、 報告架構規劃 (Report Structure)

請嚴格遵守 PDF 中的 `Grading: (1) Written Report` 章節要求。以下是詳細的各章節撰寫建議：

#### a. Title Page (封面)
*   **內容：** NCHU EE 1141 4269 Deep Learning Term Project、題目（Single Image Reflection Removal Comparison）、組員姓名、學號、具體分工（誰跑哪個模型、誰寫哪部分）。

#### b. Introduction (主題簡介)
*   **背景：** 簡述什麼是 SIRR (Single Image Reflection Removal)。解釋為什麼這很難 (Ill-posed problem, $I = T + R$)。
*   **動機：** 為什麼需要用 Deep Learning？傳統方法（Optimization-based）的限制為何？
*   **本文貢獻：** 簡單說明你們做了什麼（例如：我們復現並重新訓練了 4-6 個 SOTA 模型，並在 7 個資料集上進行了詳盡的定量與定性比較）。

#### c. Method Description (方法敘述)
*這是展現你們有讀懂論文的關鍵部分。針對你們選用的每個模型（至少 4 個，建議包含 [3]-[6]），分別撰寫一個小節。*
*   **Model [3] Revisiting (CVPR 2024):** 強調其「MaxRF」濾波器與「Location-aware」的兩階段架構。
*   **Model [4] DSIT (NeurIPS 2024):** 強調「Dual-Stream Transformer」與「Dual-Attention」機制，如何同時捕捉 Global 和 Local 特徵。
*   **Model [5] RDNet (CVPR 2025):** 強調「可逆神經網絡 (Reversible Network)」與「Prompt Generator」如何解決資訊丟失問題。
*   **Model [6] DExNet (TPAMI 2025):** 強調「Deep Unfolding」與「Sparse Coding」的結合，以及其輕量化 (Lightweight) 的特性。
*   *(若有做 Extra Credit [1] ERRNet 或 [2] KD-based，也要簡短介紹)*。
*   **Training Details (重要！):** 描述你們如何進行「Retraining」。使用了什麼 Loss function？Optimizer 是 Adam 嗎？Learning rate 設定多少？訓練了多少 Epochs？硬體規格 (GPU)？這些證明你們真的有實作。

#### d. Results Comparison & Discussion (結果比較與討論)
*這是報告的核心，分數佔比最重。*
*   **Quantitative Results (定量結果):**
    *   **Table 1:** 必須完整填寫 PDF 第 2 頁的表格 (PSNR, SSIM, NCC, LMSE, LPIPS 在各 dataset 的表現)。
    *   **Table 2:** 針對 SIR2 dataset 的參數量 (Params)、運算量 (FLOPs) 與推論時間 (Run time) 比較。
*   **Qualitative Results (定性/視覺結果):**
    *   **必須包含的圖片：** PDF 第 3 頁下方指定的圖片 ID (SIR2 objects: 111, 032, 011... 等)。
    *   **排版建議：** 模仿論文格式，一列 (Row) 放同一個場景，不同欄 (Column) 放不同模型的結果 (Input | GT | Model A | Model B ...)。
*   **Discussion (討論 - 最重要！):** 不要只貼圖表。要分析：
    *   哪個模型在 PSNR 表現最好？為什麼？（例如：DSIT 因為 Transformer 架構更能捕捉全域特徵...）
    *   哪個模型在視覺上除得最乾淨？
    *   DExNet 雖然是輕量級，但表現是否與大模型相當？
    *   在 "Wild" (野外) 資料集上，大家的表現是否都下降了？為什麼？
    *   模型 [3] 的 Location-aware 機制是否真的幫助定位反光？

#### e. Conclusion (心得及結論)
*   總結哪個模型是本次實驗的 MVP（綜合效能最好）。
*   實作過程中的困難（例如：資料集處理、訓練很久、OOM 等）。
*   未來的改進方向（例如：嘗試更新的 Transformer 架構、增加更多真實數據）。

#### f. References (參考文獻)
*   列出講義中的 8 篇論文引用格式。

---

### 二、 分工建議 (Task Assignment)

假設一組有 3-4 人，建議如下分配以達到最高效率：

#### 成員 A：主筆與理論分析 (The Writer & Theorist)
*   **負責章節：** b. Introduction, c. Method Description, e. Conclusion。
*   **任務：** 快速閱讀那 4-6 篇論文的 Abstract 和 Method 章節，歸納出核心創新點。負責撰寫文字邏輯，確保報告看起來像一篇學術文章。

#### 成員 B：實驗數據與視覺化 (The Data Analyst)
*   **負責章節：** d. Results (Table 1, Table 2)。
*   **任務：**
    *   整理所有訓練好的模型數據，填入 Table 1 和 Table 2。
    *   計算 Table 2 的 FLOPs 和 Params (講義提示可參考 DExNet GitHub)。
    *   **關鍵任務：** 去 Test dataset 裡找出 PDF 指定的那幾張圖片 (如 `SIR2 objects: 111`)，跑出每個模型的結果圖，並裁切好重點區域 (Zoom-in patches) 供成員 C 排版。

#### 成員 C：排版與總結討論 (The Editor & Discussant)
*   **負責章節：** a. Title, f. References, 以及 d. Results 的 **Discussion** 文字部分，加上全篇排版。
*   **任務：**
    *   將成員 B 產出的數據轉化為文字分析（ex: "從 Table 1 可見，RDNet 在 SSIM 上高出 0.5..."）。
    *   負責將圖片排版成精美的對比圖（參考論文 Fig. 4 的格式）。
    *   **格式檢查：** 確保字體是 Times New Roman/標楷體, 12pts, 單行間距, 邊界 2cm（這很重要，格式不對會被扣分）。
    *   整合所有人的文字與圖片，輸出最終 Word/PDF。

---

### 三、 加分與注意事項 (Tips for High Score)

1.  **針對指定圖片做深入分析：**
    PDF 特別指定了 `SIR2 objects: 111` 等圖片。在報告中，不要只是貼圖，要寫出：「在圖片 111 中，我們觀察到 [3] 模型成功移除了邊緣的反光，但 [6] 模型保留了更多背景細節...」。這代表你們真的有看結果。

2.  **強調 Training Process：**
    PDF 要求 *"You have to retrain the model"*。在報告中放一張 **Training Loss Curve** 的比較圖（如果有的話），可以強烈證明你們是自己訓練的，而不是只用 Pre-trained weights。

3.  **Table 2 的計算：**
    PDF 提示 Table 2 的計算方式可參考 [DExNet GitHub](https://github.com/jjhuangcs/DExNet)。務必去該連結找計算 FLOPs 的 script，不要自己亂算。

4.  **Extra Credit：**
    如果你們行有餘力，把 [1] ERRNet 或 [2] KD-based 的結果也放進 Table 1 比較，並在 Discussion 提到：「雖然 [1] 是 2019 的方法，但在某些指標上仍具競爭力...」，這樣可以穩拿 Extra Credit。

5.  **提交檢查：**
    最後繳交時，記得將 Code, Checkpoint, ReadMe 打包。ReadMe 寫清楚如何用你們的 code 跑出那幾張指定圖片的結果，助教改作業會很開心。

祝你們報告撰寫順利，拿高分！