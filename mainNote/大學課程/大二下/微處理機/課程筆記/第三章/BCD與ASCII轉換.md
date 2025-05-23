# BCD 與 ASCII 轉換

本節介紹 BCD (Binary Coded Decimal) 數字表示法以及 BCD 與 ASCII 之間的轉換方法。

## BCD 數字表示法

BCD (Binary Coded Decimal) 是一種使用二進制編碼表示十進制數字的方法。每個十進制數字 (0-9) 使用 4 位元二進制表示。

### BCD 編碼表

| 十進制 | BCD (4 位元) | 十六進制 |
|--------|--------------|----------|
| 0      | 0000         | 0x0      |
| 1      | 0001         | 0x1      |
| 2      | 0010         | 0x2      |
| 3      | 0011         | 0x3      |
| 4      | 0100         | 0x4      |
| 5      | 0101         | 0x5      |
| 6      | 0110         | 0x6      |
| 7      | 0111         | 0x7      |
| 8      | 1000         | 0x8      |
| 9      | 1001         | 0x9      |

### BCD 的類型

BCD 有兩種主要類型：

1. **未封裝 BCD (Unpacked BCD)**：
   - 每個位元組儲存一個 BCD 數字 (低 4 位)
   - 高 4 位通常為 0
   - 例如：數字 5 表示為 0x05

2. **封裝 BCD (Packed BCD)**：
   - 每個位元組儲存兩個 BCD 數字
   - 高 4 位和低 4 位各儲存一個數字
   - 例如：數字 25 表示為 0x25

### BCD 的優缺點

**優點**：
- 易於轉換為人類可讀的十進制
- 避免了二進制轉十進制的舍入誤差
- 適合需要精確十進制表示的應用 (如金融計算)

**缺點**：
- 儲存效率低 (比純二進制表示需要更多空間)
- 算術運算較複雜
- ARM 沒有原生 BCD 運算指令

## ASCII 數字表示

ASCII (American Standard Code for Information Interchange) 是一種字元編碼標準，用於表示文字和符號。

ASCII 數字 '0' 到 '9' 的十六進制值為 0x30 到 0x39。

| ASCII 字元 | 十六進制 | 二進制     |
|------------|----------|------------|
| '0'        | 0x30     | 0011 0000  |
| '1'        | 0x31     | 0011 0001  |
| '2'        | 0x32     | 0011 0010  |
| '3'        | 0x33     | 0011 0011  |
| '4'        | 0x34     | 0011 0100  |
| '5'        | 0x35     | 0011 0101  |
| '6'        | 0x36     | 0011 0110  |
| '7'        | 0x37     | 0011 0111  |
| '8'        | 0x38     | 0011 1000  |
| '9'        | 0x39     | 0011 1001  |

注意 ASCII 數字的高 4 位固定為 0x3，低 4 位對應數字值。

## BCD 與 ASCII 之間的轉換

### ASCII 轉 Unpacked BCD

將 ASCII 數字轉換為 Unpacked BCD 只需清除高 4 位：

```assembly
; R0 包含 ASCII 數字 (如 '5' = 0x35)
AND R0, R0, #0x0F   ; 清除高 4 位，保留低 4 位
; 現在 R0 包含 Unpacked BCD (0x05)
```

### Unpacked BCD 轉 ASCII

將 Unpacked BCD 轉換為 ASCII 數字只需設置高 4 位為 0x3：

```assembly
; R0 包含 Unpacked BCD (如 5 = 0x05)
ORR R0, R0, #0x30   ; 設置高 4 位為 0x3
; 現在 R0 包含 ASCII 數字 (0x35 = '5')
```

### Unpacked BCD 轉 Packed BCD

將兩個 Unpacked BCD 數字合併為一個 Packed BCD：

```assembly
; R0 包含第一個 Unpacked BCD (如 2 = 0x02)
; R1 包含第二個 Unpacked BCD (如 5 = 0x05)
LSL R0, R0, #4      ; 左移 4 位 (0x20)
ORR R0, R0, R1      ; 合併兩個數字
; 現在 R0 包含 Packed BCD (0x25)
```

### Packed BCD 轉 Unpacked BCD

將 Packed BCD 拆分為兩個 Unpacked BCD：

```assembly
; R0 包含 Packed BCD (如 0x25)
AND R1, R0, #0x0F   ; 提取低 4 位 (5)
LSR R0, R0, #4      ; 右移 4 位
AND R0, R0, #0x0F   ; 提取原高 4 位 (2)
; 現在 R0 包含第一個數字 (0x02)，R1 包含第二個數字 (0x05)
```

### Packed BCD 轉 ASCII

將 Packed BCD 中的兩個數字轉換為兩個 ASCII 字元：

```assembly
; R0 包含 Packed BCD (如 0x25)
; 提取並轉換高 4 位
LSR R1, R0, #4      ; 右移 4 位
ORR R1, R1, #0x30   ; 轉換為 ASCII (0x32 = '2')

; 提取並轉換低 4 位
AND R2, R0, #0x0F   ; 提取低 4 位
ORR R2, R2, #0x30   ; 轉換為 ASCII (0x35 = '5')

; 現在 R1 包含第一個 ASCII 字元，R2 包含第二個 ASCII 字元
```

### ASCII 轉 Packed BCD

將兩個 ASCII 字元轉換為一個 Packed BCD：

```assembly
; R0 包含第一個 ASCII 字元 (如 '2' = 0x32)
; R1 包含第二個 ASCII 字元 (如 '5' = 0x35)
AND R0, R0, #0x0F   ; 提取第一個數字 (2)
AND R1, R1, #0x0F   ; 提取第二個數字 (5)
LSL R0, R0, #4      ; 左移第一個數字 (0x20)
ORR R0, R0, R1      ; 合併兩個數字
; 現在 R0 包含 Packed BCD (0x25)
```

## BCD 算術運算

ARM 沒有原生 BCD 算術指令，但可以通過軟體實現：

### BCD 加法

```assembly
; R0 和 R1 包含 Unpacked BCD 數字
ADD R0, R0, R1      ; 二進制加法
CMP R0, #10         ; 檢查是否需要調整
SUBHS R0, R0, #10   ; 如果 >= 10，減去 10
MOVHS R2, #1        ; 設置進位
MOVLO R2, #0        ; 清除進位
; 現在 R0 包含 BCD 結果，R2 包含進位
```

### BCD 減法

```assembly
; R0 和 R1 包含 Unpacked BCD 數字
SUB R0, R0, R1      ; 二進制減法
CMP R0, #0          ; 檢查是否為負數
ADDMI R0, R0, #10   ; 如果 < 0，加上 10
MOVMI R2, #1        ; 設置借位
MOVPL R2, #0        ; 清除借位
; 現在 R0 包含 BCD 結果，R2 包含借位
```

## 實際應用範例

### 顯示二進制數字為 ASCII

```assembly
; 將 R0 中的二進制數字 (0-99) 轉換為兩個 ASCII 字元
; 除以 10 獲取十位數
MOV R1, R0          ; 複製原始值
MOV R2, #10         ; 除數
UDIV R0, R1, R2     ; R0 = R1 / 10 (十位數)
MUL R3, R0, R2      ; R3 = R0 * 10
SUB R4, R1, R3      ; R4 = R1 - R3 (個位數)

; 轉換為 ASCII
ORR R0, R0, #0x30   ; 十位數轉 ASCII
ORR R4, R4, #0x30   ; 個位數轉 ASCII

; 現在 R0 包含十位數的 ASCII，R4 包含個位數的 ASCII
```

### 從 ASCII 字串轉換為二進制數字

```assembly
; 假設 R0 指向包含兩個 ASCII 數字的字串
LDRB R1, [R0]       ; 載入第一個字元 (十位數)
LDRB R2, [R0, #1]   ; 載入第二個字元 (個位數)

; 轉換為二進制
SUB R1, R1, #0x30   ; ASCII 轉數字
SUB R2, R2, #0x30   ; ASCII 轉數字

; 計算最終值
MOV R3, #10
MUL R1, R1, R3      ; 十位數 * 10
ADD R0, R1, R2      ; 最終二進制值

; 現在 R0 包含轉換後的二進制數字
```

## 返回

- [[ARM算術與邏輯指令|返回第三章目錄]]
- [[微處理機概述|返回微處理機概述]]
