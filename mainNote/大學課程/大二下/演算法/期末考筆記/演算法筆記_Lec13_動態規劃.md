---
banner: 模板/banner.jpg
banner_y: "87.5"
---
# 演算法筆記 第 13 講：動態規劃 (Dynamic Programming, DP)

[[演算法筆記_總覽與考試重點]]

## 動態規劃基本概念

### 什麼是動態規劃？

-   **發明者:** Richard Bellman (1950年代)。"Programming" 在此處指的是"規劃"或"決策制定"。
-   **核心思想:** 將一個複雜問題分解成若干個重疊的子問題 (Overlapping Subproblems)，通過解決這些子問題並存儲它們的解 (避免重複計算)，最終得到原問題的解。
-   **適用條件:**
    1.  **最優子結構 (Optimal Substructure):** 問題的最優解包含其子問題的最優解。即，如果我們能找到子問題的最優解，那麼將這些子問題的最優解組合起來就能得到原問題的最優解。
    2.  **重疊子問題 (Overlapping Subproblems):** 在遞迴求解過程中，許多相同的子問題會被多次計算。DP 通過存儲子問題的解 (通常使用表格或備忘錄) 來避免這種重複計算。

### 與分治法的比較

-   **相似之處:** 都將問題分解成子問題。
-   **不同之處:**
    -   **分治法 (Divide and Conquer):** 子問題通常是**獨立的**，即它們之間不重疊 (例如，合併排序中左右兩半的排序是獨立的)。
    -   **動態規劃 (Dynamic Programming):** 子問題是**重疊的**，一個子問題的解可能被多次用於解決其他不同的父問題。

### 實現方法

1.  **帶備忘錄的自頂向下法 (Top-Down with Memoization):**
    -   使用遞迴來解決問題，與普通的遞迴解法類似。
    -   在函式開始時，檢查是否已經計算過該子問題的解 (通常查詢一個表格或哈希表)。
    -   如果已計算，直接返回存儲的解。
    -   如果未計算，則計算子問題的解，將解存儲起來，然後返回解。
    -   直觀，易於思考。

2.  **自底向上法 (Bottom-Up / Tabulation Method):**
    -   首先解決最小的子問題。
    -   然後利用這些小子問題的解，逐步解決更大的子問題，直到解決原問題。
    -   通常使用迭代方式，填寫一個表格 (DP table)。
    -   避免了遞迴的開銷，有時空間效率更高。

### 動態規劃設計步驟

1.  **刻劃最優解的結構 (Characterize the structure of an optimal solution):**
    -   描述一個最優解是如何由子問題的最優解構成的。
2.  **遞迴地定義最優解的值 (Recursively define the value of an optimal solution):**
    -   寫出一個遞迴關係式 (狀態轉移方程)，根據子問題的解來計算當前問題的解。
3.  **計算最優解的值 (Compute the value of an optimal solution):**
    -   通常採用自底向上的方式，填寫 DP 表格。
    -   或者採用帶備忘錄的自頂向下方式。
4.  **(可選) 建構最優解 (Construct an optimal solution from computed information):**
    -   在計算過程中保存額外資訊 (例如，導致最優值的選擇)，以便回溯找到實際的最優解方案，而不僅僅是最優值。

## 範例：費氏數列 (Fibonacci Numbers)

-   定義: $F(0)=0, F(1)=1, F(n) = F(n-1) + F(n-2)$ for $n \ge 2$。
-   **普通遞迴:**
    ```java
    int fib(int n) {
        if (n <= 1) return n;
        return fib(n-1) + fib(n-2); // 大量重疊子問題，指數級時間複雜度 O(2^n)
    }
    ```
-   **帶備忘錄的自頂向下 DP:**
    ```java
    int[] memo;
    int fib_memo(int n) {
        if (n <= 1) return n;
        if (memo[n] != -1) return memo[n]; // -1 表示未計算
        memo[n] = fib_memo(n-1) + fib_memo(n-2);
        return memo[n];
    }
    // 初始化 memo 陣列所有元素為 -1
    // 時間複雜度: O(n) (每個 F(i) 只計算一次)
    // 空間複雜度: O(n) (遞迴堆疊深度 + memo 陣列)
    ```
-   **自底向上 DP (Tabulation):**
    ```java
    int fib_tab(int n) {
        if (n <= 1) return n;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
    // 時間複雜度: O(n)
    // 空間複雜度: O(n) (dp 陣列)。可以優化到 O(1) 空間，因為只需要前兩個值。
    ```

## 矩陣鏈相乘問題 (Matrix Chain Multiplication) [[演算法筆記_總覽與考試重點]]

### 問題描述

給定一個 $n$ 個矩陣的序列 (鏈) $A_1, A_2, \dots, A_n$，其中矩陣 $A_i$ 的維度為 $p_{i-1} \times p_i$。我們希望計算這個矩陣鏈的乘積 $A_1 A_2 \dots A_n$。由於矩陣乘法滿足結合律，我們可以通過不同的加括號方式來計算這個乘積，而不同的加括號方式會導致不同的總純量乘法次數。目標是找到一種加括號方式，使得總的純量乘法次數最少。

-   **兩個矩陣相乘的成本:** 矩陣 $A (p \times q)$ 和 $B (q \times r)$ 相乘得到 $C (p \times r)$，需要的純量乘法次數為 $p \times q \times r$。

### 範例

考慮 $A_1 (30 \times 1)$, $A_2 (1 \times 40)$, $A_3 (40 \times 10)$, $A_4 (10 \times 25)$。
-   $((A_1 A_2) A_3) A_4$:
    -   $A_{12} = A_1 A_2$: $(30 \times 1 \times 40) = 1200$ 次乘法，結果為 $30 \times 40$ 的矩陣。
    -   $A_{123} = (A_{12}) A_3$: $(30 \times 40 \times 10) = 12000$ 次乘法，結果為 $30 \times 10$ 的矩陣。
    -   $A_{1234} = (A_{123}) A_4$: $(30 \times 10 \times 25) = 7500$ 次乘法。
    -   總計: $1200 + 12000 + 7500 = 20700$ 次。
-   $A_1 (A_2 (A_3 A_4))$:
    -   $A_{34} = A_3 A_4$: $(40 \times 10 \times 25) = 10000$ 次乘法，結果為 $40 \times 25$ 的矩陣。
    -   $A_{234} = A_2 A_{34}$: $(1 \times 40 \times 25) = 1000$ 次乘法，結果為 $1 \times 25$ 的矩陣。
    -   $A_{1234} = A_1 A_{234}$: $(30 \times 1 \times 25) = 750$ 次乘法。
    -   總計: $10000 + 1000 + 750 = 11750$ 次。
-   $A_1 ((A_2 A_3) A_4)$:
    -   $A_{23} = A_2 A_3$: $(1 \times 40 \times 10) = 400$ 次乘法，結果為 $1 \times 10$ 的矩陣。
    -   $A_{234} = (A_{23}) A_4$: $(1 \times 10 \times 25) = 250$ 次乘法，結果為 $1 \times 25$ 的矩陣。
    -   $A_{1234} = A_1 A_{234}$: $(30 \times 1 \times 25) = 750$ 次乘法。
    -   總計: $400 + 250 + 750 = 1400$ 次。 (此為最優)

### 動態規劃解法

1.  **最優子結構的刻劃:**
    考慮計算 $A_i A_{i+1} \dots A_j$ (簡記為 $A_{i..j}$) 的最優加括號方式。如果最後一次乘法是在 $A_k$ 和 $A_{k+1}$ 之間進行的，即 $(A_i \dots A_k)(A_{k+1} \dots A_j)$，那麼 $A_{i..k}$ 和 $A_{k+1..j}$ 的加括號方式也必須是它們各自子問題的最優方式。

2.  **遞迴地定義最優解的值:**
    設 $m[i, j]$ 為計算 $A_i A_{i+1} \dots A_j$ 所需的最少純量乘法次數。
    -   如果 $i = j$，則 $A_{i..j}$ 只有一個矩陣 $A_i$，不需要乘法，所以 $m[i, i] = 0$。
    -   如果 $i < j$，我們需要選擇一個分割點 $k$ ($i \le k < j$)，將鏈分成 $A_{i..k}$ 和 $A_{k+1..j}$。
        計算 $A_{i..k}$ 的成本是 $m[i, k]$。
        計算 $A_{k+1..j}$ 的成本是 $m[k+1, j]$。
        將這兩個結果矩陣相乘的成本是 $p_{i-1} \times p_k \times p_j$ (其中 $A_{i..k}$ 的維度是 $p_{i-1} \times p_k$， $A_{k+1..j}$ 的維度是 $p_k \times p_j$)。
        所以，遞迴關係式為：
        $m[i, j] = \min_{i \le k < j} \{ m[i, k] + m[k+1, j] + p_{i-1} p_k p_j \}$

3.  **計算最優解的值 (自底向上):**
    -   使用一個二維表格 `m[n][n]` 來存儲 $m[i, j]$ 的值。
    -   矩陣鏈的長度 $L$ 從 2 增加到 $n$。
    -   對於每個長度 $L$：
        -   遍歷所有可能的起始索引 $i$ (從 1 到 $n-L+1$)。
        -   計算結束索引 $j = i+L-1$。
        -   使用遞迴關係式計算 $m[i, j]$，嘗試所有可能的分割點 $k$ (從 $i$ 到 $j-1$)。
    -   同時，可以使用另一個表格 `s[n][n]` 來存儲導致 $m[i, j]$ 最優值的分割點 $k$。

    ```
    // p 是維度陣列，p[i] 是 A_i 的列數，也是 A_{i+1} 的行數。
    // A_i 的維度是 p[i-1] x p[i]。
    // 矩陣 A_1, ..., A_n，維度陣列 p 的長度是 n+1 (p[0]...p[n])

    Matrix-Chain-Order(p)
        n = p.length - 1
        let m[1..n, 1..n] and s[1..n-1, 2..n] be new tables
        for i = 1 to n
            m[i,i] = 0
        for L = 2 to n  // L is the chain length
            for i = 1 to n - L + 1
                j = i + L - 1
                m[i,j] = infinity
                for k = i to j - 1
                    q = m[i,k] + m[k+1,j] + p[i-1]*p[k]*p[j]
                    if q < m[i,j]
                        m[i,j] = q
                        s[i,j] = k // 記錄分割點
        return m and s
    ```
    -   **時間複雜度:** 三層嵌套迴圈 (L, i, k)。$L$ 從 2 到 $n$ ($O(n)$)，$i$ 從 1 到 $n-L+1$ ($O(n)$)，$k$ 從 $i$ 到 $j-1$ ($O(L)$ 或 $O(n)$)。所以總時間複雜度為 $O(n^3)$。
    -   **空間複雜度:** $O(n^2)$ (用於存儲 `m` 和 `s` 表格)。

4.  **建構最優解 (打印括號):**
    可以使用 `s` 表格遞迴地打印出最優的加括號方式。
    ```
    Print-Optimal-Parens(s, i, j)
        if i == j
            print "A" + i
        else
            print "("
            Print-Optimal-Parens(s, i, s[i,j])
            Print-Optimal-Parens(s, s[i,j] + 1, j)
            print ")"
    ```

這個問題完美地展示了動態規劃的兩個核心特徵：最優子結構和重疊子問題。直接的遞迴解法會是指數級的，而 DP 將其優化到多項式時間。