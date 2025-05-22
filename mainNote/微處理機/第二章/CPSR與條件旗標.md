# CPSR 與條件旗標

## 目前程式狀態暫存器 (CPSR)

CPSR (Current Program Status Register) 是 ARM 處理器中的一個特殊 32 位元暫存器，用於儲存處理器的狀態資訊。CPSR 包含多種資訊，其中最重要的是條件旗標 (Conditional Flags)。

## 條件旗標

CPSR 中的條件旗標位於高位元，主要包括四個旗標：

### N (Negative) 旗標

- 位於 CPSR 的第 31 位 (最高位)
- 當運算結果為負數時設置 (等於結果的最高位)
- 用於有號數比較

### Z (Zero) 旗標

- 位於 CPSR 的第 30 位
- 當運算結果為零時設置
- 用於相等性比較

### C (Carry) 旗標

- 位於 CPSR 的第 29 位
- 在以下情況設置：
  - 無號數加法產生進位
  - 無號數減法無借位
  - 移位操作中移出的位元為 1
- 用於無號數比較和多精度算術

### V (Overflow) 旗標

- 位於 CPSR 的第 28 位
- 當有號數運算結果超出範圍時設置
- 例如：兩個正數相加得到負數，或兩個負數相加得到正數
- 用於有號數比較

## 旗標更新

並非所有指令都會更新條件旗標。ARM 指令的旗標更新規則如下：

### 資料處理指令

- 預設情況下，資料處理指令 (如 ADD, SUB, MOV) 不更新旗標
- 加上 `S` 後綴 (如 `ADDS`, `SUBS`, `MOVS`) 才會更新旗標

### 比較指令

- `CMP Rn, Op2`：比較 `Rn` 和 `Op2` (`Rn - Op2`)，只更新旗標，不儲存結果
- `CMN Rn, Op2`：比較 `Rn` 和 `-Op2` (`Rn + Op2`)，只更新旗標，不儲存結果
- `TST Rn, Op2`：測試 `Rn AND Op2` 的結果，只更新旗標
- `TEQ Rn, Op2`：測試 `Rn EOR Op2` 的結果，只更新旗標
- 這些指令總是更新旗標

### 邏輯指令

邏輯指令 (AND, ORR, EOR, BIC, MVN) 加上 `S` 後綴會更新 N, Z, C 旗標 (V 不受影響)。

### 分支指令

分支指令 (B, BL) 不影響旗標。

## 存取 CPSR

可以使用特殊指令直接存取 CPSR：

```assembly
MRS Rd, CPSR    ; 讀取 CPSR 到 Rd
MSR CPSR, Rm    ; 寫入 Rm 到 CPSR
```

但在一般程式設計中，很少需要直接存取 CPSR。通常是透過條件執行或條件分支來使用旗標。

## 條件碼

ARM 指令集支援條件執行，可以在大多數指令後加上條件碼，使指令只在特定條件滿足時才執行。

### 無號數條件碼

- **EQ** (Equal)：Z=1，相等
- **NE** (Not Equal)：Z=0，不相等
- **CS/HS** (Carry Set/Higher or Same)：C=1，無借位/大於等於
- **CC/LO** (Carry Clear/Lower)：C=0，有借位/小於
- **HI** (Higher)：C=1 且 Z=0，大於
- **LS** (Lower or Same)：C=0 或 Z=1，小於等於

### 有號數條件碼

- **MI** (Minus)：N=1，負數
- **PL** (Plus)：N=0，正數或零
- **VS** (Overflow Set)：V=1，溢位
- **VC** (Overflow Clear)：V=0，無溢位
- **GT** (Greater Than)：Z=0 且 N=V，大於
- **LT** (Less Than)：N≠V，小於
- **GE** (Greater or Equal)：N=V，大於等於
- **LE** (Less or Equal)：Z=1 或 N≠V，小於等於

### 條件執行範例

```assembly
CMP R0, R1        ; 比較 R0 和 R1
ADDGT R2, R2, #1  ; 如果 R0 > R1，則 R2 = R2 + 1
MOVLE R3, #0      ; 如果 R0 <= R1，則 R3 = 0
```

## 條件分支

條件分支指令根據旗標狀態決定是否跳轉：

```assembly
CMP R0, #10       ; 比較 R0 和 10
BEQ Equal         ; 如果 R0 = 10，跳轉到 Equal
BGT Greater       ; 如果 R0 > 10，跳轉到 Greater
B Default         ; 無條件跳轉到 Default

Equal:
    ; R0 = 10 的處理程式碼
    ...

Greater:
    ; R0 > 10 的處理程式碼
    ...

Default:
    ; 其他情況的處理程式碼
    ...
```

## 旗標在實際應用中的使用

### 迴圈控制

```assembly
MOV R0, #10       ; 初始化計數器
Loop:
    ; 迴圈主體
    ...
    SUBS R0, R0, #1  ; 遞減計數器並更新旗標
    BNE Loop         ; 如果 R0 != 0，繼續迴圈
```

### 條件執行

```assembly
CMP R0, #0        ; 檢查 R0 是否為 0
MOVEQ R1, #1      ; 如果 R0 = 0，則 R1 = 1
MOVNE R1, #2      ; 如果 R0 != 0，則 R1 = 2
```

### 多重條件

```assembly
CMP R0, #10       ; 比較 R0 和 10
BLT Less          ; 如果 R0 < 10，跳轉到 Less
CMP R0, #20       ; 比較 R0 和 20
BLE LessOrEqual   ; 如果 R0 <= 20，跳轉到 LessOrEqual
; R0 > 20 的處理程式碼
...
```

## 返回

- [[ARM架構與組合語言程式設計|返回第二章目錄]]
- [[微處理機概述|返回微處理機概述]]
