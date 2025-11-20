---
banner: 模板/banner.jpg
banner_y: "93.5"
---
### MobileNetV2 物件偵測模型的量化感知訓練 (基於 TF2 2.12)

這份筆記為物件偵測模型量化流程的練習：**訓練一個浮點模型 -> 進行量化感知訓練 (Quantization-Aware Training, QAT) -> 轉換為最終的 uint8 TFLite 模型**。

選擇 **MobileNetV2** 作為骨幹網路，並在其上搭建一個簡單的物件偵測模型，使用 **PASCAL VOC 2012** 資料集進行遷移訓練。

---

### 第 1 部分 : 環境設定 (TF 2.12 + Conda)

經過多次嘗試，發現 TensorFlow/Keras 從版本 2.16 開始的重大 API 變更，與生態系統中的其他庫（如 `tensorflow-model-optimization`）存在嚴重的相容性問題。

為了確保能夠順利進行量化流程，我決定採用被廣泛驗證、相容性最佳的配置來避掉相容性問題：**TensorFlow 2.12**。這份筆記的方案會使用 Conda/Mamba 來管理完全獨立 GPU 環境，避免與系統級的 CUDA 驅動產生衝突，要不然很可能因為 TF 降版本就與系統 CUDA 衝突。

#### 1.1 前置條件

*   一個可以正常工作的 WSL2 (Ubuntu) 環境。
*   在 **Windows 主機** 上已安裝 NVIDIA 顯卡驅動。
*   已安裝 Mambaforge 或 Miniconda。本筆記會使用 `mamba` 

#### 1.2 建立 `tf_mobilenet` 虛擬環境

建立一個名為 `tf_mobilenet` 的全新環境，並在其中安裝所有版本精確匹配的依賴。

1.  **建立並啟用新環境**
    ```bash
	# 建立一個包含 Python 3.10 的新環境
	mamba create -n tf_mobilenet python=3.10 -y
	
	# 啟用新環境
	mamba activate tf_mobilenet
    ```

#### 1.3 安裝所有依賴

分兩步，結合使用 mamba 和 Pip 來建構相容環境。

1.  **使用 Mamba 安裝核心依賴**
    使用 Mamba 來安裝 TensorFlow 以及與之配套的 CUDA Toolkit 和 cuDNN。Conda 會將這些庫安裝在虛擬環境內部，與系統環境完全隔離。
2. **-c nvidia** ( 在前 ) : 優先從 nvidia 官方頻道安裝 CUDA Toolkit 跟  cuDNN

    ```bash
	# 使用 -c conda-forge 來確保獲取到 GPU 版本的 TensorFlow
	# Conda 會自動解析並安裝與 TF 2.12 相容的 cudatoolkit 和 cudnn
	mamba install -c nvidia -c conda-forge "tensorflow-gpu=2.12.*" "cudatoolkit=11.8.*" matplotlib opencv lxml tqdm -y
    ```

3.  **使用 Pip 安裝 TFMOT**
    在核心依賴安裝完成後，使用 Pip 來安裝與 TF 2.12 相容的 `tensorflow-model-optimization` 版本。

    ```bash
	# 安裝與 TF 2.12 配套的 TFMOT 0.7.5 版本
	pip install "tensorflow-model-optimization==0.7.5"
    ```

#### 1.4 最終環境驗證

完成以上所有步驟後， `tf_mobilenet` 環境就準備就緒。執行以下指令進行驗證：

```bash
python -c "import tensorflow as tf; print(f'TensorFlow Version: {tf.__version__}'); print(tf.config.list_physical_devices('GPU'))"
```

如果一切順利，會看到如下輸出，代表開發環境已就緒：
```bash
TensorFlow Version: 2.12.x
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

### 第 2 部分 : 資料準備 (PASCAL VOC 2012 from Kaggle)

原始的 PASCAL VOC 資料集連結經常無法存取，我改用更穩定可靠的 Kaggle 作為資料來源。此部分將使用 Kaggle API 下載資料，並將其整理成專案腳本所需的標準格式。

#### 2.1 首次設定：配置 Kaggle API

如果是第一次在當前環境中使用 Kaggle API，請先完成 API 設定。

1.  **安裝 Kaggle API 套件**
    ```bash
	pip install kaggle
    ```

2.  **獲取並配置 API Token**
    *   登入的 [Kaggle 網站](https://www.kaggle.com/)，點擊頭像進入 **"Account"**，在 API 區塊點擊 **"Create New Token"** 以下載 `kaggle.json`。
    *   將下載的 `kaggle.json` 檔案移動到 WSL 環境中的正確位置，並設定權限：
    ```bash
	# 建立 .kaggle 資料夾 (如果不存在)
	mkdir -p ~/.kaggle
	# 假設檔案在 Windows 的 "下載" 資料夾中
	mv /mnt/c/Users/<Your-Windows-Username>/Downloads/kaggle.json ~/.kaggle/
	# 設定正確的檔案權限
	chmod 600 ~/.kaggle/kaggle.json
    ```

#### 2.2 下載並解壓縮資料集

1.  **下載資料**
    在專案目錄下，執行以下指令下載資料集：
    ```bash
kaggle datasets download -d gopalbhattrai/pascal-voc-2012-dataset
    ```

2.  **解壓縮檔案**
    ```bash
	# (如果需要) 安裝 unzip 工具
	sudo apt-get update && sudo apt-get install -y unzip
	# 解壓縮
	unzip pascal-voc-2012-dataset.zip
	```
    解壓縮後，會得到 `VOC2012_train_val/` 和 `VOC2012_test/` 兩個資料夾。

#### 2.3 使用腳本自動合併與整理

為了避免手動複製貼上多行指令可能發生的錯誤，將所有整理步驟打包成一個自動化腳本較為便捷

1.  **建立腳本檔案**
    在專案根目錄下，建立一個名為 `organize_voc_data.sh` 的腳本檔案

2.  **貼上腳本內容**
    將下方的完整程式碼區塊複製並貼到編輯器中：
    ```bash
	#!/bin/bash
	
	# 如果任何指令失敗，立即停止
	set -e
	
	echo ">>> 1. 建立最終目標目錄..."
	mkdir -p VOCdevkit/VOC2012
	
	echo ">>> 2. 使用 rsync 從深層路徑合併 train_val 內容 (這會建立初始目錄)..."
	rsync -av --progress VOC2012_train_val/VOC2012_train_val/ VOCdevkit/VOC2012/
	
	echo ">>> 3. 使用 rsync 從深層路徑合併 test 內容 (這會合併到現有目錄)..."
	rsync -av --progress VOC2012_test/VOC2012_test/ VOCdevkit/VOC2012/
	
	echo ">>> 4. 清理所有不再需要的原始資料夾..."
	rm -rf VOC2012_train_val/
	rm -rf VOC2012_test/
	
	echo ">>> 資料整理完成！"
	echo ">>> 驗證最終結構："
	ls -F VOCdevkit/VOC2012/
    ```

1.  **儲存並執行腳本**
    *   接著，賦予腳本執行權限：
        ```bash
	chmod +x organize_voc_data.sh
        ```
    *   最後，執行腳本來自動完成所有整理工作：
        ```bash
	./organize_voc_data.sh
        ```

腳本執行完畢後，專案目錄結構應該如下：
```bash
Quantization_practice/
├── VOCdevkit/
│   └── VOC2012/
│       ├── Annotations/
│       ├── ImageSets/
│       ├── JPEGImages/
│       └── ...
│
├── organize_voc_data.sh
└── ... (其他專案檔案)
```
可以執行 `ls VOCdevkit/VOC2012/ImageSets/Main/` 來確認 `test.txt` 等檔案是否被正確合併。

#### 2.4 建立資料載入腳本

編寫一個 Python 腳本 (`data_loader.py`) 來處理資料載入相關 API

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
    
    # 為了簡化，只取第一個物件進行訓練 ( PASCAL VOC 中有些會有同一影像多物件，只取第一個該圖像最明顯物件進行偵測 )
    # 完整的物件偵測可能會需要更複雜的標籤格式，為了簡化量化流程與複雜度，我這邊只實作單物件偵測
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
為了簡化流程，這個資料載入器會將多物件偵測問題簡化為**單物件偵測**（只取第一個標註的物件）。

---

### 第 3 部分 : 模型建立與浮點訓練

在這一部分，將建立物件偵測模型並進行標準的浮點數訓練。有個關鍵是**解除模型巢狀結構**，建構一個單一、扁平的 Keras Functional 模型，不然之後量化可能會出現問題。

在執行前，請先建立 `models` 和 `logs` 資料夾：`mkdir models logs`。

#### 3.1 建立模型 (`model.py`)

**`model.py`**
```python
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.models import Model
from data_loader import VOC_CLASSES

def build_detection_model(input_shape=(224, 224, 3)):
    # 載入預訓練的 MobileNetV2 進行遷移學習
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, # 不包含頂部分類層，因是做物件偵測
        weights='imagenet' # imagenet 預訓練權重
    )
    # 凍結骨幹網絡的權重
    base_model.trainable = False

    # 直接使用 base_model 的輸入當成 input
    inputs = base_model.input

    # 從 base_model 的輸出開始連接我們自己的層 ( 物件偵測 --> bbox 跟 分類 )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 邊界框回歸頭
    bbox_output = Dense(4, activation='sigmoid', name='bbox_head')(x)

    # 類別分類頭
    num_classes = len(VOC_CLASSES)
    class_output = Dense(num_classes, activation='softmax', name='class_head')(x)

    # 創建一個新的 Model，它的輸入是 base_model 的輸入，輸出是我們自己設定的物件偵測輸出
    model = Model(inputs=inputs, outputs=[bbox_output, class_output])

    return model
```

#### 3.2 開始浮點訓練 (`train_float.py`)

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
        'bbox_head': 'mse', # 這裡使用均方誤差來回歸邊界框
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
            'models/float_model_temp.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
)

print("Floating-point training finished.")
# 保存最後的模型
model.save('models/float_model.h5')
```
**執行訓練:**
```bash
python train_float.py
```
訓練完成後，會在 `models` 資料夾下得到一個名為 `float_model_final.h5` 的浮點模型。

---

### 第 4 部分 : 量化感知訓練 (QAT)

載入訓練好的浮點模型，並使用 `tensorflow-model-optimization` (TFMOT) 工具對其進行量化感知微調。

**`train_qat.py`**
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from data_loader import get_dataset

# --- 設定參數 ---
DATA_DIR = 'VOCdevkit/VOC2012'
BATCH_SIZE = 32
QAT_EPOCHS = 5
QAT_LEARNING_RATE = 1e-5

# --- 載入資料 ---
print("Loading datasets for QAT...")
train_dataset = get_dataset(DATA_DIR, 'train', BATCH_SIZE)
val_dataset = get_dataset(DATA_DIR, 'val', BATCH_SIZE)

# --- 載入浮點模型並應用 QAT ---
print("Loading floating-point model...")

base_model = tf.keras.models.load_model('models/float_model.h5')
print("Model loaded successfully.")

# 使用 QAT API
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(base_model)

# 編譯 QAT 模型
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE),
    # 使用 TFMOT 重命名後的新層名作為鍵值
    loss={
        'quant_bbox_head': 'mse',
        'quant_class_head': 'sparse_categorical_crossentropy'
    },
    metrics={
        'quant_class_head': 'accuracy'
    }
)

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
**執行 QAT 訓練:**
```bash
python train_qat.py
```
訓練完成後，會得到 `models/qat_model.h5`。

---

### 第 5 部分：轉換為 TFLite (uint8)

將 QAT 模型轉換為最終部署用的 `uint8` TFLite 格式。

**`convert_to_tflite.py`**
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# 創建一個包含了所有 TFMOT 自定義對象的 scope
quantize_scope = tfmot.quantization.keras.quantize_scope

# 在 scope 內加載模型
with quantize_scope():
    qat_model = tf.keras.models.load_model('models/qat_model.h5')

# 創建 TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)

# 設定最佳化選項，這會啟用 uint8 量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 轉換時，TFLiteConverter 會需要一個代表性的數據集來確定量化參數（min/max ranges)
# 從 data_loader 中獲取一小部分數據
from data_loader import get_dataset

def representative_dataset_gen():
    # 取一個 batch 的數據
    for image, _ in get_dataset('VOCdevkit/VOC2012', 'val', batch_size=1).take(1):
        yield [image]

converter.representative_dataset = representative_dataset_gen
# 確保轉換器強制使用整數（uint8）量化
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8


# 進行轉換
tflite_quant_model = converter.convert()

# 保存 TFLite 模型
with open('models/mobilenet_detect_uint8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Successfully converted QAT model to uint8 TFLite model.")
print("Saved to models/mobilenet_detect_uint8.tflite")
```
**執行轉換：**
```bash
python convert_to_tflite.py
```

---

### 第 6 部分：使用 TFLite 模型進行推論

最後，驗證一下這個量化後的模型是否能正常運作。

**`inference.py`**
```python
import tensorflow as tf
import numpy as np
import cv2
import time
from data_loader import VOC_CLASSES

# --- 設定參數 ---
MODEL_PATH = 'models/mobilenet_detect_uint8.tflite'
IMAGE_PATH = 'VOCdevkit/VOC2012/JPEGImages/2008_000031.jpg'
CONFIDENCE_THRESHOLD = 0.5

# --- 載入 TFLite 模型並分配張量 ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# --- 取得輸入和輸出張量的詳細資訊 ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 準備輸入影像 ---
img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_shape = input_details[0]['shape']
h, w = input_shape[1], input_shape[2]
img_resized = cv2.resize(img_rgb, (w, h))
input_data = np.expand_dims(img_resized, axis=0)
if input_details[0]['dtype'] == np.uint8:
    input_data = input_data.astype(np.uint8)

# --- 執行推論 ---
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# 根據 Keras 模型的輸出順序直接獲取張量
# outputs=[bbox_output, class_output] -> 0: bbox, 1: class
bbox_output_detail = output_details[0]
class_output_detail = output_details[1]

raw_bbox_output = interpreter.get_tensor(bbox_output_detail['index'])
raw_class_output = interpreter.get_tensor(class_output_detail['index'])

# --- 对输出进行反量化 ---
# 获取 class head 的反量化参数
class_scale, class_zero_point = class_output_detail['quantization']
class_output_float = class_scale * (raw_class_output.astype(np.float32) - class_zero_point)

# 获取 bbox head 的反量化参数
bbox_scale, bbox_zero_point = bbox_output_detail['quantization']
bbox_output_float = bbox_scale * (raw_bbox_output.astype(np.float32) - bbox_zero_point)

# --- 後處理並視覺化 ---
pred_label_idx = np.argmax(class_output_float[0])
pred_confidence = class_output_float[0][pred_label_idx]
pred_class = VOC_CLASSES[pred_label_idx]

print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
print(f"Prediction Class: {pred_class}")
print(f"Confidence: {pred_confidence:.4f}")

if pred_confidence > CONFIDENCE_THRESHOLD and pred_class != 'background':
    h_orig, w_orig, _ = img.shape
    box = bbox_output_float[0]
    
    if np.any(box < 0) or np.any(box > 1):
        print("Warning: Bounding box coordinates are out of [0, 1] range. Check model training.")
    else:
        ymin = int(box[0] * h_orig)
        xmin = int(box[1] * w_orig)
        ymax = int(box[2] * h_orig)
        xmax = int(box[3] * w_orig)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f'{pred_class}: {pred_confidence:.2f}'
        cv2.putText(img, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print("Bounding box drawn.")
else:
    print(f"Prediction ignored. Confidence ({pred_confidence:.2f}) is below threshold ({CONFIDENCE_THRESHOLD}) or class is 'background'.")

cv2.imwrite('output.jpg', img)
print("Result saved to output.jpg")
```
**執行推論：**
```bash
python inference.py
```
讀取一張測試圖片，使用的 `uint8` TFLite 模型進行偵測，並將結果繪製在圖片上，儲存為 `output.jpg`。
