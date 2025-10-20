---
banner: 模板/banner.jpg
banner_y: "88.5"
---
好的，這是一份根據您提供的 PDF 內容，整理而成的詳細 Markdown 步驟教學文件。

---

# YOLOv3 訓練與量化部署實戰教學 (基於 TensorFlow)

這份教學將引導您從零開始，完成從環境建置、資料準備、模型訓練，到最終將模型量化並部署的完整流程。核心目標是產出一個輕量化、高效能、適合在邊緣運算裝置上運行的 `uint8` 整數模型。

## 🎯 教學地圖

1.  **步驟一：建立開發環境** - 安裝所有必要的軟體與函式庫。
2.  **步驟二：準備訓練資料集** - 將您的影像資料處理成模型可讀取的格式。
3.  **步驟三：模型訓練** - 進行標準的浮點數訓練與進階的量化感知訓練。
4.  **步驟四：模型推論與部署** - 將訓練好的模型轉換為 TFLite 格式並進行驗證。

---

## 步驟一：建立開發環境

這是所有工作的基礎，請務必確保每個環節都安裝正確。

### 1.1 安裝 Anaconda

Anaconda 是一個 Python 環境管理器，可以幫助我們輕鬆管理不同專案所需的套件版本。

- **Windows**:
    1.  前往 [Anaconda 官網](https://www.anaconda.com/products/distribution) 下載 Windows 版本的安裝程式 (.exe)。
    2.  執行安裝程式，在 "Advanced Options" 步驟中，**不建議**勾選 "Add Anaconda to my PATH environment variable"。保持預設即可。
    3.  安裝完成後，從開始功能表打開 "Anaconda Prompt" 進入指令環境。

- **Ubuntu**:
    1.  前往官網下載 Linux 版本的安裝腳本 (.sh)。
    2.  打開終端機 (Terminal)，執行安裝腳本：
        ```bash
        bash ./Anaconda3-xxxx.xx-Linux-x86_64.sh
        ```
    3.  在安裝過程中，當詢問是否要將 Anaconda 路徑加入 `.bashrc` 時，請輸入 `yes`。
    4.  安裝完成後，重開一個新的終端機視窗，環境變數即會生效。

### 1.2 安裝 NVIDIA CUDA 與 cuDNN

為了使用 GPU 進行高速訓練，必須安裝這兩項工具。

⚠️ **重要提醒**：CUDA/cuDNN 的版本**必須**與您要安裝的 TensorFlow 版本相容！請先查詢 [TensorFlow 官方文件](https://www.tensorflow.org/install/source#gpu) 確認版本對應關係。

1.  **安裝 CUDA Toolkit**:
    - 前往 [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 頁面。
    - 選擇符合您 TensorFlow 需求的版本 (例如 `9.0`, `10.1`, `11.2` 等)。
    - 根據您的作業系統 (Windows/Linux) 下載並安裝。

2.  **安裝 cuDNN**:
    - 前往 [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) 頁面 (需要註冊 NVIDIA 開發者帳號)。
    - 下載與您 CUDA 版本對應的 cuDNN 函式庫。
    - 解壓縮後，將 `bin`, `include`, `lib` 三個資料夾中的檔案，分別複製到 CUDA 的安裝路徑下對應的資料夾中。

### 1.3 安裝 TensorFlow (GPU 版本)

1.  打開 Anaconda Prompt (Windows) 或終端機 (Ubuntu)。
2.  (建議) 建立一個獨立的 conda 環境：
    ```bash
    conda create -n yolov3_tf python=3.7
    conda activate yolov3_tf
    ```
3.  安裝 TensorFlow GPU 版本：
    ```bash
    pip install tensorflow-gpu==<your_required_version> # 例如 1.14 或 2.x
    ```
4.  驗證安裝是否成功：
    ```python
    import tensorflow as tf
    print(tf.test.is_gpu_available())
    # 如果顯示 True，代表 GPU 環境設定成功
    ```

### 1.4 (可選) 實用工具

- **TensorBoard**: 訓練過程視覺化工具，通常隨著 TensorFlow 一併安裝。
- **Netron**: 模型結構視覺化工具，可查看 `.pb` 或 `.tflite` 檔案的架構。可至 [Netron GitHub](https://github.com/lutzroeder/netron/releases) 下載桌面應用程式。

---

## 步驟二：準備訓練資料集

這裡以 `Pascal VOC` 資料集格式為例。

### 2.1 下載資料集

- 從 [PASCAL VOC 官網](http://host.robots.ox.ac.uk/pascal/VOC/) 或鏡像站下載資料集，並解壓縮。您會得到 `VOCdevkit` 資料夾。

### 2.2 提取特定類別資料 (可選)

如果您的專案只需要偵測 VOC 資料集中的某幾類物件（例如：只想偵測人、車、貓），可以使用 `extract_XX.py` 腳本來篩選。

1.  打開 `extract_07.py` 或 `extract_12.py`。
2.  修改 `classes` 變數，只留下您需要的類別名稱。
3.  修改檔案路徑，指向您的原始資料集位置與您想儲存篩選後資料的位置。
4.  執行腳本：`python extract_07.py`。

### 2.3 產生訓練/驗證清單

使用 `voc2txt.py` 腳本，將資料集隨機劃分為訓練集、驗證集和測試集。

1.  修改 `xmlfilepath` 指向您標籤檔 (.xml) 所在的路徑。
2.  執行腳本：`python voc2txt.py`。
3.  腳本會在指定路徑下生成 `train.txt`, `val.txt`, `test.txt` 等檔案，裡面記錄了影像的檔名。

### 2.4 轉換為模型輸入格式

最後，使用 `parse_voc_xml.py` 腳本，將 XML 標籤檔和上一步生成的清單轉換成模型訓練程式 (`train.py`) 真正需要的格式。

1.  修改 `voc_07` 或 `voc_12` 的路徑，指向您的資料集根目錄。
2.  執行腳本：`python misc/parse_voc_xml.py`。
3.  這會生成最終的 `train.txt` 和 `val.txt`，格式如下：
    ```
    影像路徑 影像寬 影像高 物件類別ID x_min y_min x_max y_max 物件類別ID x_min ...
    ```

---

## 步驟三：模型訓練

訓練分為兩個主要階段：先進行標準的浮點數訓練，待模型收斂後，再進行量化感知訓練。

### 3.1 浮點數訓練 (Floating-to-Floating) - 標準流程

💡 **目標**：訓練一個高精度的 32 位元浮點 (FP32) 模型。

1.  **下載預訓練權重**：從專案提供的連結下載 `darknet` 的預訓練權重 (`.weights` 檔)，並透過 `convert_weight.py` 轉換為 TensorFlow 的 `.ckpt` 格式。這能大幅加速訓練收斂。
2.  **開始訓練**：執行訓練腳本。
    ```bash
    python train.py
    ```
3.  **監控訓練**：在另一個終端機視窗中，啟動 TensorBoard 來觀察 Loss 和 mAP 的變化。
    ```bash
    tensorboard --logdir ./data/logs
    ```
4.  持續訓練直到模型的 mAP 達到滿意的水平。

### 3.2 量化感知訓練 (Quantization-Aware Training) - 進階流程

💡 **目標**：在 FP32 模型的基礎上，模擬量化過程中的精度損失，讓模型提前適應，最終得到一個對量化「不敏感」的模型。

1.  **載入模型**：將上一步訓練好的 FP32 模型權重 (`.ckpt` 檔) 作為預訓練模型。
2.  **修改訓練代碼**：在 `train.py` 中，找到定義 `loss` 和 `optimizer` 的地方，在兩者之間插入量化訓練 API。
    ```python
    # 原代碼
    loss = yolo_model.compute_loss(...)
    # ...
    optimizer = tf.train.AdamOptimizer(...)
    train_op = optimizer.minimize(loss)

    # 修改後
    loss = yolo_model.compute_loss(...)
    
    # --- 插入量化API ---
    g = tf.get_default_graph()
    tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)
    # --------------------

    optimizer = tf.train.AdamOptimizer(...)
    train_op = optimizer.minimize(loss)
    ```
    > `quant_delay` 是一個計步器，設為 0 表示從一開始就模擬量化。

3.  **開始訓練**：使用較小的學習率 (learning rate) 進行微調 (fine-tuning)。
    ```bash
    python train.py --learning_rate 1e-5
    ```
4.  訓練完成後，您會得到一個已經「準備好」被量化的模型 (`.ckpt` 檔)。

---

## 步驟四：模型推論與部署

將訓練好的模型轉換為最終的 TFLite 格式。

### 4.1 凍結圖譜 (Freeze Graph)，生成 `.pb` 檔

此步驟會將模型的架構和訓練好的權重合併成一個檔案。

1.  在推論腳本 (`eval.py` 或 `inference.py`) 中，載入量化感知訓練後的 `.ckpt` 檔。
2.  使用 `tf.graph_util.convert_variables_to_constants` 函數將圖譜凍結。
3.  將凍結後的圖譜寫入 `.pb` 檔案。

### 4.2 轉換為 TFLite (uint8) 模型

這是最關鍵的一步，使用 `toco` (TensorFlow Lite Converter) 工具進行轉換。

```bash
toco tflite_convert \
  --graph_def_file=./yolov3.pb \
  --output_file=./yolov3_uint8.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLite \
  --input_shape="1,416,416,3" \
  --input_array=input_data \
  --output_array=bbox_1,bbox_2,bbox_3 \
  --inference_type=QUANTIZED_UINT8 \
  --mean_values=0 \
  --std_dev_values=255 \
  --default_ranges_min=0 \
  --default_ranges_max=6
```

**重要參數說明**:
| 參數 | 說明 |
| :--- | :--- |
| `graph_def_file` | 輸入的 `.pb` 檔路徑。 |
| `output_file` | 輸出的 `.tflite` 檔路徑。 |
| `input_shape` | 模型輸入的形狀 (batch, height, width, channel)。 |
| `input_array` | 輸入節點的名稱。 |
| `output_array` | 輸出節點的名稱 (多個用逗號分隔)。 |
| `inference_type` | 指定轉換類型為 **`QUANTIZED_UINT8`**。 |
| `mean_values` | 輸入資料的均值。如果訓練時已將像素值縮放到 [0, 1]，此處通常為 `128`。若無，則為 `0`。 |
| `std_dev_values` | 輸入資料的標準差。若縮放到 [0, 1]，此處為 `128`。若原始 [0, 255]，則為 `255`。 |
| `default_ranges...` | 為圖中沒有範圍資訊的激活層 (如 ReLU6) 提供預設的量化範圍。 |

### 4.3 執行推論 (Inference)

使用 `tflite-inference.py` 腳本來測試您生成的 `.tflite` 模型。

```bash
python tflite-inference.py --model ./yolov3_uint8.tflite --img ./test.jpg
```
您可以比較量化後 (`uint8`) 和量化前 (`float`) 的推論結果，會發現準確率非常接近，但 `uint8` 模型的運算速度在支援的硬體上會快很多。

### 4.4 (進階) 產生硬體驗證資料

如果您正在開發硬體加速器，可能需要模型中間層的輸出結果來進行比對。

- **技巧**：在執行 `toco` 轉換時，將您想觀察的中間層節點名稱，也加入到 `--output_array` 參數中。
- **範例**：
  ```bash
  --output_array=bbox_1,yolov3/darknet53_body/Conv_1/Relu
  ```
- 這樣生成的 `.tflite` 模型就會有兩個輸出。在推論時，您就可以同時取得最終結果和 `Conv_1/Relu` 層的輸出值，存成 `.npy` 檔供硬體驗證使用。

## 總結

恭喜！您已經走完了從環境建置到部署一個高效能 YOLOv3 模型的完整旅程。這份教學的核心價值在於詳細介紹了**量化感知訓練**，這項技術是實現 AI 模型在資源受限的硬體上高效運行的關鍵。