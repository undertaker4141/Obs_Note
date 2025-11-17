---
banner: 模板/banner.jpg
banner_y: "87"
---
好的，這份教學將會帶領您使用最新的工具鏈（Mamba, TensorFlow 2.x）來重現您提供的 PDF 中的核心流程：**訓練一個浮點模型 -> 進行量化感知訓練 (Quantization-Aware Training, QAT) -> 轉換為最終的 uint8 TFLite 模型**。

我們將選擇 **MobileNetV2** 作為骨幹網絡，並在其上搭建一個簡單的物件偵測模型，使用經典的 **PASCAL VOC** 資料集進行訓練，這與原 PDF 的流程最為貼近。

您的硬體環境 (`RTX 4070`, `CUDA 12.5`) 非常先進，最新的 TensorFlow 將能完美支援。

---

### MobileNetV2 物件偵測模型的量化感知訓練 (基於 TF2)

這份教學將詳細說明從環境建置到最終模型推論的每一步。

#### 第 1 部分：環境設定 (使用 Mamba)

Mamba 是一個比 Conda 更快的套件管理器，其指令與 Conda 高度相容。

1.  **安裝 Mambaforge**
    如果您尚未安裝，請先從 [Mambaforge GitHub Releases](https://github.com/conda-forge/miniforge/releases) 下載並安裝 Mambaforge。它會提供一個最小化的 Conda 環境，並以 Mamba 作為預設的套件管理器。

2.  **建立並啟用虛擬環境**
    打開您的 WSL2 Ubuntu 終端機，執行以下指令來建立一個名為 `tf_mobilenet` 的虛擬環境。

    ```bash
    # 建立一個包含 Python 3.10 的環境 (TF 官方推薦)
    mamba create -n tf_mobilenet python=3.10 -y

    # 啟用環境
    mamba activate tf_mobilenet
    ```

3.  **安裝必要的 Python 套件**
    我們將安裝 TensorFlow、TensorFlow 模型最佳化工具 (TFMOT)、以及其他輔助工具。

    ```bash
    # 安裝 TensorFlow (會自動選擇與您 CUDA 12.5 相容的版本)
    pip install tensorflow

    # 安裝 TFMOT (用於量化感知訓練)
    pip install tensorflow-model-optimization

    # 安裝其他輔助工具
    pip install matplotlib opencv-python "lxml>=4.9.0" tqdm
    ```
    *   `lxml`: 用於解析 PASCAL VOC 的 XML 標註檔。
    *   `opencv-python`: 用於影像處理。
    *   `matplotlib`: 用於視覺化。
    *   `tqdm`: 用於顯示進度條。

4.  **環境驗證**
    確認 TensorFlow 可以成功偵測到您的 RTX 4070 GPU。

    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```
    如果一切順利，您應該會看到類似以下的輸出，代表 GPU 已成功被 TensorFlow 抓到：
    ```
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    ```

#### 第 2 部分：資料準備 (PASCAL VOC 2012)

與原 PDF 流程一致，我們使用 PASCAL VOC 資料集。

1.  **下載資料集**
    在您的專案目錄下，執行以下指令下載並解壓縮 VOC 2012 訓練/驗證資料集。

    ```bash
    # 下載資料集
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

    # 解壓縮
    tar -xvf VOCtrainval_11-May-2012.tar
    ```
    解壓縮後，您會得到一個名為 `VOCdevkit` 的資料夾。

2.  **建立資料載入腳本**
    TensorFlow 2.x 推薦使用 `tf.data.Dataset` API 來建立高效的資料載入管線。我們將編寫一個 Python 腳本 (`data_loader.py`) 來處理這一切。

    **`data_loader.py`**
    ```python
    import tensorflow as tf
    import os
    import xml.etree.ElementTree as ET
    import numpy as np

    # PASCAL VOC 的 20 個類別 + 1 個背景類別
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]

    # 創建一個類別名稱到索引的映射
    CLASS_MAP = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    def parse_xml_annotation(xml_path):
        """解析 PASCAL VOC 的 XML 標註檔"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # 排除困難或被截斷的物件
            difficult = int(obj.find('difficult').text)
            truncated = int(obj.find('truncated').text)
            if difficult or truncated:
                continue

            label_name = obj.find('name').text
            labels.append(CLASS_MAP[label_name])
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

    def load_and_preprocess_image(image_path, boxes, labels, target_size=(224, 224)):
        """載入並預處理影像和邊界框"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = tf.shape(image)
        
        # 影像歸一化
        image = tf.image.resize(image, target_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        # 調整邊界框座標
        h_ratio = target_size[0] / tf.cast(original_shape[0], tf.float32)
        w_ratio = target_size[1] / tf.cast(original_shape[1], tf.float32)
        
        # [xmin, ymin, xmax, ymax] -> [ymin, xmin, ymax, xmax] for tf.image
        boxes_tf = tf.cast(boxes, tf.float32)
        y1 = boxes_tf[:, 1:2] * h_ratio
        x1 = boxes_tf[:, 0:1] * w_ratio
        y2 = boxes_tf[:, 3:4] * h_ratio
        x2 = boxes_tf[:, 2:3] * w_ratio
        
        # 為了簡化，我們只取第一個物件進行訓練
        # 完整的物件偵測需要更複雜的標籤格式 (如 SSD 的 anchor)
        # 這裡為了演示量化流程，我們簡化為單物件偵測
        if tf.shape(boxes)[0] > 0:
            box = tf.concat([y1[0], x1[0], y2[0], x2[0]], axis=0) / target_size[0] # 正規化到 0-1
            label = labels[0]
        else:
            # 如果影像中沒有物件，使用一個假的標籤
            box = tf.zeros((4,), dtype=tf.float32)
            label = 0 # background

        return image, (box, label)

    def get_dataset(data_dir, split='train', batch_size=32):
        """創建 tf.data.Dataset"""
        img_dir = os.path.join(data_dir, 'JPEGImages')
        anno_dir = os.path.join(data_dir, 'Annotations')
        split_file = os.path.join(data_dir, 'ImageSets', 'Main', f'{split}.txt')
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
            
        image_paths = [os.path.join(img_dir, f'{img_id}.jpg') for img_id in image_ids]
        anno_paths = [os.path.join(anno_dir, f'{img_id}.xml') for img_id in image_ids]

        # 創建一個 Dataset 物件
        path_ds = tf.data.Dataset.from_tensor_slices((image_paths, anno_paths))
        
        def generator():
            for img_path, anno_path in zip(image_paths, anno_paths):
                boxes, labels = parse_xml_annotation(anno_path)
                # 跳過沒有物件的影像
                if len(labels) > 0:
                    yield img_path, boxes, labels

        # 使用 from_generator 處理變長資料
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )

        dataset = dataset.map(
            lambda img_path, boxes, labels: load_and_preprocess_image(img_path, boxes, labels),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    ```
    **注意**: 為了簡化教學，這個資料載入器將多物件偵測問題簡化為**單物件偵測**（只取第一個標註的物件）。這足以展示從訓練到量化的完整流程。

#### 第 3 部分：模型建立與浮點訓練

1.  **建立模型**
    我們使用 Keras Functional API 建立模型。骨幹是預訓練的 MobileNetV2，頂部接上我們自訂的分類頭和邊界框回歸頭。

    **`model.py`**
    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
    from tensorflow.keras.models import Model
    from data_loader import VOC_CLASSES # 引用類別數量

    def build_detection_model(input_shape=(224, 224, 3)):
        # 載入預訓練的 MobileNetV2，不包含頂部分類層
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # 凍結骨幹網路的權重
        base_model.trainable = False

        # 自訂模型頭
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        
        # 邊界框回歸頭 (4 個輸出: ymin, xmin, ymax, xmax)
        bbox_output = Dense(4, activation='sigmoid', name='bbox_head')(x)
        
        # 類別分類頭
        num_classes = len(VOC_CLASSES)
        class_output = Dense(num_classes, activation='softmax', name='class_head')(x)
        
        model = Model(inputs=inputs, outputs=[bbox_output, class_output])
        return model
    ```

2.  **開始浮點訓練**
    建立一個主訓練腳本 `train_float.py`。

    **`train_float.py`**
    ```python
    import tensorflow as tf
    from model import build_detection_model
    from data_loader import get_dataset

    # --- 設定參數 ---
    DATA_DIR = 'VOCdevkit/VOC2012'
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # --- 載入資料 ---
    print("Loading datasets...")
    train_dataset = get_dataset(DATA_DIR, 'train', BATCH_SIZE)
    val_dataset = get_dataset(DATA_DIR, 'val', BATCH_SIZE)
    print("Datasets loaded.")

    # --- 建立並編譯模型 ---
    model = build_detection_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'bbox_head': 'mse', # 使用均方誤差來回歸邊界框
            'class_head': 'sparse_categorical_crossentropy'
        },
        metrics={
            'class_head': 'accuracy'
        }
    )
    model.summary()

    # --- 開始訓練 ---
    print("Starting floating-point training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir='./logs/float'),
            tf.keras.callbacks.ModelCheckpoint(
                'models/float_model.h5', 
                save_best_only=True, 
                monitor='val_loss'
            )
        ]
    )

    print("Floating-point training finished.")
    # 保存最後的模型
    model.save('models/float_model_final.h5')
    ```
    在執行前，請先建立 `models` 和 `logs` 資料夾：`mkdir models logs`。
    現在，執行訓練：
    ```bash
    python train_float.py
    ```
    訓練完成後，您會在 `models` 資料夾下得到一個名為 `float_model.h5` 的浮點模型。

#### 第 4 部分：量化感知訓練 (QAT)

這是將模型轉換為量化版本的關鍵步驟，對應原 PDF 中的 `tf.contrib.quantize` 流程，但在 TF2 中我們使用 TFMOT。

1.  **建立 QAT 訓練腳本**
    我們將載入剛剛訓練好的浮點模型，並使用 TFMOT 對其進行量化感知微調。

    **`train_qat.py`**
    ```python
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    from data_loader import get_dataset

    # --- 設定參數 ---
    DATA_DIR = 'VOCdevkit/VOC2012'
    BATCH_SIZE = 32
    QAT_EPOCHS = 5 # 量化微調通常不需要太多 epoch
    QAT_LEARNING_RATE = 1e-5 # 使用非常小的學習率

    # --- 載入資料 ---
    print("Loading datasets for QAT...")
    train_dataset = get_dataset(DATA_DIR, 'train', BATCH_SIZE)
    val_dataset = get_dataset(DATA_DIR, 'val', BATCH_SIZE)

    # --- 載入浮點模型並應用 QAT ---
    quantize_model = tfmot.quantization.keras.quantize_model

    # 載入之前訓練好的浮點模型
    base_model = tf.keras.models.load_model('models/float_model_final.h5')

    # 將 QAT 應用於整個模型
    qat_model = quantize_model(base_model)

    # 編譯 QAT 模型
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE),
        loss={
            'bbox_head': 'mse',
            'class_head': 'sparse_categorical_crossentropy'
        },
        metrics={
            'class_head': 'accuracy'
        }
    )
    qat_model.summary()

    # --- 開始 QAT 微調 ---
    print("Starting Quantization-Aware Training...")
    history = qat_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=QAT_EPOCHS,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir='./logs/qat')
        ]
    )

    print("QAT finished.")
    # 保存 QAT 模型
    qat_model.save('models/qat_model.h5')
    ```
    執行 QAT 訓練：
    ```bash
    python train_qat.py
    ```
    這一步會在模型中插入模擬量化的節點，並微調權重以適應量化帶來的精度損失。訓練完成後，您會得到 `models/qat_model.h5`。

#### 第 5 部分：轉換為 TFLite (uint8)

現在我們將 QAT 模型轉換為最終部署用的 `uint8` TFLite 格式。

1.  **建立轉換腳本**

    **`convert_to_tflite.py`**
    ```python
    import tensorflow as tf

    # 載入 QAT 模型
    # 必須使用 `quantize_scope` 來載入包含 TFMOT 層的模型
    with tf.keras.utils.custom_object_scope(
        {'QuantizeWrapperV2': tf.keras.layers.Layer}
    ):
        qat_model = tf.keras.models.load_model('models/qat_model.h5')

    # 創建 TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    
    # 設定最佳化選項，這會啟用 uint8 量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 進行轉換
    tflite_quant_model = converter.convert()

    # 保存 TFLite 模型
    with open('models/mobilenet_detect_uint8.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    print("Successfully converted QAT model to uint8 TFLite model.")
    print("Saved to models/mobilenet_detect_uint8.tflite")
    ```
    執行轉換：
    ```bash
    python convert_to_tflite.py
    ```
    恭喜！您現在已經在 `models` 資料夾中得到了最終的 `mobilenet_detect_uint8.tflite` 模型。

#### 第 6 部分：使用 TFLite 模型進行推論

最後，我們來驗證一下這個量化後的模型是否能正常工作。

**`inference.py`**
```python
import tensorflow as tf
import numpy as np
import cv2
import time
from data_loader import VOC_CLASSES # 引用類別名稱

# --- 載入 TFLite 模型並分配張量 ---
interpreter = tf.lite.Interpreter(model_path='models/mobilenet_detect_uint8.tflite')
interpreter.allocate_tensors()

# --- 取得輸入和輸出張量的詳細資訊 ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 準備輸入影像 ---
# 找一張測試圖片，例如從 VOC 資料集中挑選
image_path = 'VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg' 
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 預處理影像 (resize + 轉為 uint8)
# 注意：uint8 模型的輸入不需要手動做-1到1的歸一化，TFLite內部會處理
input_shape = input_details[0]['shape']
h, w = input_shape[1], input_shape[2]
img_resized = cv2.resize(img_rgb, (w, h))
input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

# --- 執行推論 ---
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# --- 取得輸出結果 ---
# 輸出的順序可能與模型定義時不同，需要根據 `output_details` 確認
# 假設 output[0] 是 bbox, output[1] 是 class
if 'bbox_head' in output_details[0]['name']:
    bbox_output = interpreter.get_tensor(output_details[0]['index'])
    class_output = interpreter.get_tensor(output_details[1]['index'])
else:
    bbox_output = interpreter.get_tensor(output_details[1]['index'])
    class_output = interpreter.get_tensor(output_details[0]['index'])
    
# --- 後處理並視覺化 ---
pred_label_idx = np.argmax(class_output[0])
pred_confidence = class_output[0][pred_label_idx]
pred_class = VOC_CLASSES[pred_label_idx]

# 將正規化的 bbox 座標還原
h_orig, w_orig, _ = img.shape
box = bbox_output[0]
ymin = int(box[0] * h_orig)
xmin = int(box[1] * w_orig)
ymax = int(box[2] * h_orig)
xmax = int(box[3] * w_orig)

# 在原圖上繪製結果
cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
label_text = f'{pred_class}: {pred_confidence:.2f}'
cv2.putText(img, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
print(f"Prediction: {label_text}")

# 儲存或顯示結果
cv2.imwrite('output.jpg', img)
print("Result saved to output.jpg")
```
執行推論：
```bash
python inference.py
```
這將會讀取一張測試圖片，使用您的 `uint8` TFLite 模型進行偵測，並將結果繪製在圖片上，保存為 `output.jpg`。

### 總結

您已經成功地使用最新的 TensorFlow 2.x 和 TFMOT 工具鏈，完成了一個從零開始的量化感知訓練流程：
1.  **環境建置**：使用 Mamba 建立了一個乾淨且高效的開發環境。
2.  **資料準備**：使用 `tf.data` API 建立了 PASCAL VOC 資料集的載入管線。
3.  **浮點訓練**：基於預訓練的 MobileNetV2 訓練了一個基礎的物件偵測模型。
4.  **量化感知訓練**：使用 TFMOT 對浮點模型進行微調，使其適應量化。
5.  **模型轉換**：將 QAT 模型轉換為高效能的 `uint8` TFLite 格式。
6.  **模型推論**：驗證了最終的量化模型可以成功執行。

這個流程完整地重現了原 PDF 的核心精神，並將其更新為當前業界主流的實踐方法。