# PC 相對定址

本節介紹 ARM 處理器中的 PC 相對定址 (PC-Relative Addressing) 原理和應用，包括 ADR 指令、LDR 偽指令和文字池。

## PC 相對定址概念

PC 相對定址是一種使用程式計數器 (PC, R15) 作為基底暫存器的定址模式。位址計算方式為 PC 的值加上一個偏移量。

PC 相對定址的主要優點是：
- **位置無關代碼 (Position-Independent Code, PIC)**：程式碼可以載入到記憶體的任何位置執行
- **簡化位址計算**：不需要知道絕對位址
- **減少重定位 (Relocation) 需求**：連結器可以更容易地重定位程式碼

## PC 值的特殊性

在 ARM Thumb-2 模式下，讀取 PC 時，其值通常是「目前指令位址 + 4」。這是因為 ARM 處理器的流水線設計，PC 指向的是下一條要提取的指令。

這意味著在計算 PC 相對位址時，需要考慮這個 +4 的偏移。

## PC 相對載入指令

```assembly
LDR Rd, [PC, #offset]    ; Rd = Memory[PC + offset]
```

這個指令從 PC + offset 位址載入資料到 Rd。

**範例**：
```assembly
    LDR R0, [PC, #12]    ; 從 PC+12 位址載入資料到 R0
```

## ADR 偽指令

ADR 是一個偽指令 (Pseudo-instruction)，用於計算標籤的位址並載入到暫存器：

```assembly
ADR Rd, label    ; Rd = label 的位址
```

組譯器會將 ADR 轉換為一個或多個指令，通常是 ADD 或 SUB 指令，用於計算 PC 相對位址。

**範例**：
```assembly
    ADR R0, MyData    ; R0 = MyData 的位址
    
MyData
    DCD 0x12345678
```

ADR 指令的範圍有限，通常只能存取當前程式碼區段附近的位址 (約 ±1KB)。

## LDR 偽指令 (=符號)

對於無法使用 ADR 存取的位址或大立即數，可以使用 LDR 偽指令：

```assembly
LDR Rd, =value    ; Rd = value 或 value 的位址
```

組譯器會根據 value 的性質選擇不同的實現方式：
- 如果 value 是可以用 MOV 表示的小立即數，則轉換為 MOV 指令
- 如果 value 是大立即數或標籤，則將其放入文字池 (Literal Pool)，並產生 PC 相對的 LDR 指令來載入

**範例**：
```assembly
    LDR R0, =0x12345678    ; 載入大立即數
    LDR R1, =MyArray       ; 載入陣列位址
```

## 文字池 (Literal Pool)

文字池是組譯器在程式碼區段中插入的一塊資料區域，用於儲存 LDR 偽指令使用的常數和位址。

文字池通常放在函式或程式碼區段的末尾，但可以使用 LTORG 指令強制在特定位置產生文字池：

```assembly
    LDR R0, =0x12345678    ; 使用文字池中的常數
    ; 其他程式碼...
    
    LTORG                  ; 在此處產生文字池
    
    ; 更多程式碼...
```

如果程式碼區段太長，可能需要多個文字池，因為 PC 相對定址的範圍有限 (約 ±4KB)。

## PC 相對定址的應用

### 存取常數資料

```assembly
    ADR R0, Constants    ; R0 = Constants 的位址
    LDR R1, [R0, #0]     ; R1 = 第一個常數
    LDR R2, [R0, #4]     ; R2 = 第二個常數
    
Constants
    DCD 0x12345678, 0x87654321
```

### 存取字串

```assembly
    ADR R0, HelloString    ; R0 = 字串位址
    ; 使用 R0 處理字串...
    
HelloString
    DCB "Hello, World!", 0
```

### 函數指標表 (跳轉表)

```assembly
    LDR R0, =JumpTable    ; R0 = 跳轉表位址
    LDR R1, [R0, R2, LSL #2]  ; R1 = JumpTable[R2]
    BX R1                 ; 跳轉到函數
    
JumpTable
    DCD Function1, Function2, Function3, Function4
    
Function1
    ; 函數 1 的程式碼
    BX LR
    
Function2
    ; 函數 2 的程式碼
    BX LR
    
Function3
    ; 函數 3 的程式碼
    BX LR
    
Function4
    ; 函數 4 的程式碼
    BX LR
```

### 位置無關代碼

```assembly
    ADR R0, DataStart    ; R0 = DataStart 的位址
    ADR R1, DataEnd      ; R1 = DataEnd 的位址
    SUB R2, R1, R0       ; R2 = 資料大小
    
    ; 使用 R0 和 R2 處理資料...
    
DataStart
    ; 資料開始
    DCB 1, 2, 3, 4, 5
DataEnd
    ; 資料結束
```

## PC 相對定址的限制

PC 相對定址有一些限制：

1. **範圍限制**：
   - ADR 指令的範圍約 ±1KB
   - LDR [PC, #offset] 的範圍約 ±4KB
   
2. **文字池位置**：
   - 文字池必須在 PC 相對定址範圍內
   - 長函數可能需要多個文字池
   
3. **PC 值的特殊性**：
   - 需要考慮 PC = 目前指令位址 + 4 的特性
   - 這可能導致手動計算 PC 相對偏移時出錯

## PC 相對定址與絕對定址的比較

| 特性 | PC 相對定址 | 絕對定址 |
|------|------------|---------|
| 位置無關性 | 高 | 低 |
| 範圍 | 有限 (±4KB) | 全部 4GB 位址空間 |
| 重定位需求 | 低 | 高 |
| 程式碼大小 | 通常較小 | 通常較大 |
| 執行效率 | 高 | 高 |

## 返回

- [[ARM定址模式|返回第六章目錄]]
- [[微處理機概述|返回微處理機概述]]
