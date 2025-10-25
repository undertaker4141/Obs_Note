---
banner: 模板/banner.jpg
banner_y: "87.5"
---
好的，沒有問題。我已經詳細分析了您提供的這份 PDF 教學文件。這是一份非常完整且紮實的 YOLOv3 訓練與量化流程指南，但它確實是基於現已過時的 TensorFlow 1.x 框架，特別是 `tf.contrib.quantize` 這個 API。

我將遵循這份文件的核心精神與步驟，但將所有操作都轉換為現代 TensorFlow 2.x 的實踐方法，特別是使用 **Keras API** 和 **TensorFlow Model Optimization Toolkit (TFMOT)**。這將會比原始流程更簡潔、更直觀。

以下是為您整理的 Markdown 格式的 TF2.x 版本實戰指南：

---

# YOLOv3 訓練與量化實戰指南 (TensorFlow 2.x 版本)

本指南將引導您完成使用 TensorFlow 2.x 訓練 YOLOv3 模型，並對其進行量化，以部署到輕量級邊緣裝置的完整流程。我們將參考您提供的 PDF 文件的結構，但採用 TF2.x 的現代化工具鏈。

##  Part 1: 環境設定 (Environment Setup)

設定一個乾淨的 Python 環境是成功的第一步。推薦使用 Anaconda 或 Python 的 `venv`。

1.  **安裝 Anaconda**:
    如果您尚未安裝，請從 [Anaconda 官網](https://www.anaconda.com/products/distribution) 下載並安裝。

2.  **建立 Conda 環境**:
    打開終端機 (Anaconda Prompt)，建立一個專用的環境（例如，命名為 `tf-yolo`）。

    ```bash
    conda create -n tf-yolo python=3.8
    conda activate tf-yolo
    ```

3.  **安裝 TensorFlow 與 GPU 支持**:
    確保您的 NVIDIA 驅動程式、CUDA Toolkit 和 cuDNN 版本與您要安裝的 TensorFlow 版本相容。詳細的對應版本請參考 [TensorFlow 官方說明](https://www.tensorflow.org/install/gpu)。

    ```bash
    # 安裝 TensorFlow GPU 版本
    pip install tensorflow

    # 驗證安裝
    python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
    ```
    如果輸出顯示找到 GPU 裝置，代表設定成功。

4.  **安裝必要套件**:
    除了 TensorFlow，我們還需要其他輔助套件。

    ```bash
    pip install numpy opencv-python matplotlib tqdm
    ```

## Part 2: 專案準備與模型下載

我們將使用一個成熟的 TF2.x YOLOv3 開源專案，這將為我們省去大量重寫模型架構的時間。我們採用您之前找到的 [YunYang1994](https://github.com/YunYang1994/tensorflow-yolov3) 所推薦的 TF2.0 版本。

1.  **下載專案程式碼**:
    ```bash
    git clone https://github.com/YunYang1994/TensorFlow2.0-Examples.git
    cd TensorFlow2.0-Examples/4-Object_Detection/YOLOV3
    ```
    *注意：後續所有指令都在 `YOLOV3` 這個資料夾內執行。*

2.  **下載並轉換預訓練權重**:
    為了進行微調 (fine-tuning) 或直接進行量化，我們需要官方的預訓練權重。

    ```bash
    # 下載官方 darknet 權重
    wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights

    # 執行專案提供的轉換腳本
    python load_weights.py
    ```
    執行成功後，您會在 `checkpoints` 資料夾下看到轉換好的 TF2.x 格式權重 (`.tf` 副檔名)。

## Part 3: 數據集準備 (Dataset Preparation)

此步驟與原 PDF 教學的目標一致：將您的數據集 (例如 PASCAL VOC) 轉換為 YOLO 訓練所需的格式。該專案需要一個 `train.txt` 和 `val.txt` 文件，每一行格式如下：
`圖片路徑 x_min,y_min,x_max,y_max,class_id x_min,y_min,x_max,y_max,class_id ...`

1.  **準備數據**:
    *   下載 [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 數據集並解壓縮。
    *   您會得到一個 `VOCdevkit` 資料夾。

2.  **生成標註文件**:
    專案中提供了 `scripts/voc_annotation.py` 腳本來自動生成所需的 `.txt` 文件。

    *   **修改類別**: 打開 `scripts/voc_annotation.py`，找到 `classes` 列表，您可以修改它以符合您的需求，或保留預設的 20 個類別。
    *   **執行腳本**:
        ```bash
        python scripts/voc_annotation.py --data_path /path/to/your/VOCdevkit --train_txt_path ./data/voc_train.txt --val_txt_path ./data/voc_val.txt
        ```
        請將 `/path/to/your/VOCdevkit` 換成您自己的實際路徑。

3.  **準備類別名稱文件**:
    在 `data/` 資料夾下，建立一個 `voc.names` 文件，內容是您的類別名稱，每行一個，順序必須和訓練時的 class_id 一致。

## Part 4: 浮點模型訓練 (Floating-Point Model Training)

現在我們可以開始訓練標準的浮點模型。

1.  **修改設定檔 (可選)**:
    您可以查看 `yolov3/configs.py` 文件，根據您的需求調整學習率、批次大小 (batch size) 等超參數。

2.  **開始訓練**:
    執行 `train.py` 腳本，並指定數據集文件和類別名稱文件。

    ```bash
    python train.py --dataset ./data/voc_train.txt --val_dataset ./data/voc_val.txt --classes ./data/voc.names --weights ./checkpoints/yolov3.tf
    ```
    *   `--weights` 參數指定了使用預訓練權重進行微調。
    *   訓練過程中，您可以另開一個終端機，使用 TensorBoard 監控訓練狀況：
        ```bash
        tensorboard --logdir=log
        ```

3.  **儲存模型**:
    訓練完成後，模型權重會保存在 `checkpoints` 資料夾。為了後續量化，我們需要將其轉換為 Keras 的 `.h5` 完整模型格式。

    *   您可以在 `train.py` 訓練結束後，加入 `model.save('yolov3_float.h5')` 這樣的程式碼來儲存模型。

## Part 5: 量化感知訓練 (Quantization Aware Training - QAT)

這是與 TF1.x 差異最大的地方。我們不再使用 `tf.contrib.quantize`，而是使用 **TensorFlow Model Optimization Toolkit (TFMOT)**。

1.  **安裝 TFMOT**:
    ```bash
    pip install -q tensorflow-model-optimization
    ```

2.  **建立 QAT 模型**:
    QAT 的流程是在已經訓練好的浮點模型基礎上，插入偽量化節點 (fake quantization nodes)，然後再進行短時間的微調。

    ```python
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    # 1. 載入您訓練好的浮點 Keras 模型
    float_model = tf.keras.models.load_model('checkpoints/yolov3_float.h5') # 假設您已儲存

    # 2. 使用 TFMOT API 將其轉換為 QAT 模型
    quantize_model = tfmot.quantization.keras.quantize_model(float_model)

    # 3. 編譯 QAT 模型
    #    學習率要設得非常低，因為我們只是在微調
    quantize_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                           # Loss 函數需要和您原訓練時的一致
                           loss=...) # 根據專案的 loss function 填寫

    # 4. 進行短時間的微調訓練 (例如，用一小部分數據訓練幾個 epoch)
    #    數據加載方式與 Part 4 相同
    quantize_model.fit(train_dataset, epochs=5, validation_data=val_dataset)

    # 5. 儲存量化感知訓練後的模型
    quantize_model.save('yolov3_qat.h5')
    ```
    *這段程式碼需要整合到專案的訓練流程中，或另外寫一個腳本來執行。*

## Part 6: 匯出與轉換為 TFLite (Export & Convert to TFLite)

訓練好 QAT 模型後，就可以將其轉換為最終的 `uint8` TFLite 模型。這個過程也完全在 TF2.x 中以 Python API 完成。

1.  **建立 TFLiteConverter**:
    我們從剛剛儲存的 QAT Keras 模型開始。

    ```python
    import tensorflow as tf
    import numpy as np

    # 載入 QAT 模型
    qat_model = tf.keras.models.load_model('yolov3_qat.h5')

    # 建立轉換器
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)

    # 2. 設定優化選項 (這是觸發整數量化的關鍵)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. (重要) 提供代表性數據集 (Representative Dataset)
    #    這能讓轉換器校準 (calibrate) 量化參數，對於性能至關重要
    def representative_dataset():
        # 從您的驗證集中取約 100 張圖片
        for image_path, _ in val_dataset.take(100):
            # 讀取並預處理圖片，使其 shape 和 type 符合模型輸入
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [416, 416])
            img = img / 255.
            img = np.expand_dims(img, axis=0).astype(np.float32)
            yield [img]

    converter.representative_dataset = representative_dataset
    
    # 4. 確保輸入和輸出也是整數格式 (推薦)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 5. 執行轉換
    tflite_quant_model = converter.convert()

    # 6. 儲存模型
    with open('yolov3_quant.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    print("Successfully converted and saved the quantized TFLite model.")
    ```

## Part 7: 執行 TFLite 推論 (Inference)

使用 TensorFlow Lite Interpreter 在 Python 中執行量化後的模型。

```python
import tensorflow as tf
import numpy as np
import cv2

# 1. 載入 TFLite 模型並分配 Tensors
interpreter = tf.lite.Interpreter(model_path="yolov3_quant.tflite")
interpreter.allocate_tensors()

# 2. 獲取輸入和輸出 tensor 的詳細資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. 準備輸入圖片
image_path = 'path/to/your/test_image.jpg'
input_shape = input_details[0]['shape'] # e.g., [1, 416, 416, 3]

img = cv2.imread(image_path)
img = cv2.resize(img, (input_shape[2], input_shape[1]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(img, axis=0)
# 如果輸入類型是 uint8，則不需要除以 255
# input_data = (input_data / 255.).astype(np.float32) # For float input

# 4. 設置輸入 Tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# 5. 執行推論
interpreter.invoke()

# 6. 獲取輸出 Tensor
# 根據您的模型輸出，可能會有多個 output
output_data = interpreter.get_tensor(output_details[0]['index'])
# ... process other outputs if any ...

print(output_data)
```

## Part 8: (選修) 獲取中間層輸出用於硬體驗證

原 PDF 提到為了驗證硬體 (Verilog) 設計，需要獲取中間層的輸出。在 TF2.x 中，這變得非常簡單。

**方法一：在 Keras 中建立多輸出模型**

在定義 Keras 模型時，除了最終輸出，也可以將任何您感興趣的中間層作為輸出。

```python
# 假設您在搭建 Functional API 模型
base_model = ...
x = base_model.get_layer('some_conv_layer').output
...
final_output = ...

# 建立一個同時輸出最終結果和中間層結果的模型
debug_model = tf.keras.Model(inputs=base_model.input, outputs=[final_output, x])
```

**方法二：在 TFLite Interpreter 中直接讀取**

TFLite Interpreter 允許您在 `invoke()` 之後讀取任意 tensor 的值，這非常強大。

```python
# ... 接續 Part 7 的程式碼 ...

# 獲取所有 tensor 的詳細資訊
tensor_details = interpreter.get_tensor_details()

# 找到您感興趣的層 (例如，通過名稱)
target_tensor_index = -1
for i, tensor in enumerate(tensor_details):
    if tensor['name'] == 'StatefulPartitionedCall:1': # 名字可能需要您用 Netron 查看
        target_tensor_index = i
        break

if target_tensor_index != -1:
    # 執行 invoke() 後...
    interpreter.invoke()
    
    # 直接獲取該層的輸出
    intermediate_output = interpreter.get_tensor(target_tensor_index)
    print(f"Shape of intermediate tensor: {intermediate_output.shape}")
```

---

這份指南涵蓋了從環境設定到最終量化模型推論的完整流程，並全部採用了 TensorFlow 2.x 的最佳實踐。相較於 TF1.x，您可以看到流程更加整合，程式碼也更貼近標準的 Python 開發習慣。祝您專案順利！
