---
banner: 模板/banner.jpg
banner_y: "87"
---
# 演算法筆記 第 9 講：圖形搜尋與拓樸排序

[[演算法筆記_總覽與考試重點]]

## 圖 (Graph) 的基本概念

### 無向圖 (Undirected Graphs)

-   **定義:** 圖由一組**頂點 (vertices)** (或節點 nodes) 和一組連接頂點對的**邊 (edges)** 組成。在無向圖中，邊 $(u, v)$ 和 $(v, u)$ 是相同的。
-   **術語:**
    -   **路徑 (Path):** 由邊連接的頂點序列。
    -   **環 (Cycle):** 起點和終點相同的路徑。
    -   **連通 (Connected):** 如果兩個頂點之間存在路徑，則它們是連通的。
    -   **連通分量 (Connected Component):** 一個圖中的極大連通子圖。圖中的每個頂點都屬於某個連通分量。
    -   **頂點的度 (Degree of a vertex):** 與該頂點相連的邊的數量。
-   **應用:** 社交網路、電路、交通網路等。

### 有向圖 (Directed Graphs / Digraphs)

-   **定義:** 邊是有方向的。邊 $(u, v)$ 從頂點 $u$ 指向頂點 $v$ ($u \rightarrow v$)，這與邊 $(v, u)$ 不同。
-   **術語:**
    -   **有向路徑 (Directed Path):** 遵循邊的方向的頂點序列。
    -   **有向環 (Directed Cycle):** 遵循邊的方向的環。
    -   **出度 (Out-degree):** 從一個頂點指出的邊的數量。
    -   **入度 (In-degree):** 指向一個頂點的邊的數量。
    -   **有向無環圖 (Directed Acyclic Graph, DAG):** 不包含任何有向環的有向圖。
-   **應用:** 任務排程 (先決條件)、網頁連結、流程圖等。

### 圖的表示法

1.  **邊列表 (Edge List):**
    -   直接存儲所有邊的列表，每條邊表示為一對頂點 (以及可能的權重)。
    -   空間: $O(E)$。
    -   查找與某頂點相鄰的邊效率低。
2.  **鄰接矩陣 (Adjacency Matrix):**
    -   一個 $V \times V$ 的矩陣 `adj[][]`，其中 `adj[u][v] = 1` (或權重) 如果存在邊 $(u,v)$，否則為 0 (或 $\infty$)。
    -   無向圖的鄰接矩陣是對稱的。
    -   空間: $O(V^2)$。對於稀疏圖 (邊數 $E$ 遠小於 $V^2$) 來說空間效率低。
    -   檢查兩頂點間是否有邊: $O(1)$。
    -   遍歷某頂點的所有鄰接點: $O(V)$。
3.  **鄰接列表 (Adjacency List):**
    -   一個包含 $V$ 個列表的陣列 (或列表的列表)，其中第 $v$ 個列表存儲所有與頂點 $v$ 相鄰的頂點 (或從 $v$ 指出的邊)。
    -   空間: $O(V+E)$。對於稀疏圖非常高效。
    -   遍歷某頂點的所有鄰接點: $O(\text{degree}(v))$。
    -   這是實踐中最常用的表示法。

## 圖搜尋演算法 (Graph Traversal)

系統性地訪問圖中所有可達頂點的方法。

### 深度優先搜尋 (Depth-First Search, DFS)

-   **策略:** 沿著一條路徑盡可能深地探索，直到到達末端或已訪問過的頂點，然後回溯並探索其他分支。類似於走迷宮時沿著一面牆走。
-   **實現 (遞迴):**
    ```java
    boolean[] marked; // 標記已訪問的頂點
    int[] edgeTo;     // edgeTo[w] = v 表示從 v 第一次訪問到 w

    void dfs(Graph G, int v) {
        marked[v] = true;
        // 在訪問 v 時執行某些操作
        for (int w : G.adj(v)) { // 遍歷 v 的所有鄰接點 w
            if (!marked[w]) {    // 如果 w 未被訪問
                edgeTo[w] = v;   // 記錄路徑
                dfs(G, w);       // 遞迴訪問 w
            }
        }
        // 在 v 的所有鄰接點都訪問完畢後執行某些操作 (例如後序遍歷)
    }
    ```
-   **資料結構:**
    -   `marked[]`: 布林陣列，記錄頂點是否已被訪問。
    -   `edgeTo[]`: 整數陣列，用於重建從起點到任一已訪問頂點的路徑。`edgeTo[w] = v` 表示是通過邊 `v-w` 首次到達 `w`。
    -   隱式使用函式呼叫堆疊來實現遞迴。
-   **應用:**
    -   尋找路徑。
    -   檢測環 (Cycle Detection)。
    -   尋找連通分量。
    -   拓撲排序 (用於 DAG)。
-   **時間複雜度:** $O(V+E)$，因為每個頂點和每條邊最多被訪問常數次。

### 廣度優先搜尋 (Breadth-First Search, BFS)

-   **策略:** 從源頂點開始，首先訪問所有直接相鄰的頂點，然後是距離為2的頂點，依此類推。逐層探索。
-   **實現 (使用佇列):**
    ```java
    boolean[] marked;
    int[] edgeTo;
    int[] distTo; // distTo[v] = 從起點到 v 的最短路徑長度 (邊的數量)
    Queue<Integer> queue = new LinkedList<>();

    void bfs(Graph G, int s) {
        marked[s] = true;
        distTo[s] = 0;
        queue.add(s);

        while (!queue.isEmpty()) {
            int v = queue.remove();
            // 訪問 v
            for (int w : G.adj(v)) {
                if (!marked[w]) {
                    marked[w] = true;
                    edgeTo[w] = v;
                    distTo[w] = distTo[v] + 1;
                    queue.add(w);
                }
            }
        }
    }
    ```
-   **資料結構:**
    -   `marked[]`, `edgeTo[]` 同 DFS。
    -   `distTo[]`: 記錄從源點到各點的最短路徑長度 (以邊的數量計)。
    -   佇列 (Queue): 存儲待訪問的頂點。
-   **應用:**
    -   尋找無權圖中的最短路徑 (以邊的數量計)。
    -   尋找連通分量。
-   **時間複雜度:** $O(V+E)$。

## 拓撲排序 (Topological Sort) [[演算法筆記_總覽與考試重點]]

### 定義與前提

-   **定義:** 對於一個**有向無環圖 (DAG)**，拓撲排序是其頂點的一個線性排序，使得對於圖中每條有向邊 $(u, v)$ (從 $u$ 到 $v$)，頂點 $u$ 都在頂點 $v$ 之前出現。
-   **前提:** 只有 DAG 才能進行拓撲排序。如果一個有向圖包含環，則無法進行拓撲排序。
    -   如果存在環 $v_1 \rightarrow v_2 \rightarrow \dots \rightarrow v_k \rightarrow v_1$，那麼在任何線性排序中，$v_1$ 必須在 $v_2$ 之前，$v_2$ 必須在 $v_3$ 之前 ... $v_k$ 必須在 $v_1$ 之前，這產生矛盾。
-   **應用:**
    -   任務排程：如果任務 A 必須在任務 B 之前完成，則可以將其建模為 $A \rightarrow B$ 的邊，拓撲排序給出一個可行的任務執行順序。
    -   編譯器中的依賴解析。
    -   課程先修關係。

### 基於 DFS 的拓撲排序演算法

-   **演算法:**
    1.  對 DAG 執行深度優先搜尋 (DFS)。
    2.  在 DFS 的過程中，記錄每個頂點完成訪問 (即其所有後代都已被訪問完畢，從遞迴呼叫返回時) 的順序。
    3.  拓撲排序即為頂點**完成順序的逆序** (也稱為逆後序遍歷，Reverse Postorder)。
-   **為什麼可行 (直觀解釋):**
    -   考慮一條邊 $u \rightarrow v$。
    -   當 `dfs(u)` 被呼叫時：
        -   **情況 1: `dfs(v)` 已經完成。** 這意味著 $v$ 在 $u$ 之前完成。在逆後序中，$u$ 會在 $v$ 之前。
        -   **情況 2: `dfs(v)` 尚未被呼叫。** 那麼 `dfs(u)` 會在其遞迴過程中呼叫 `dfs(v)`。因此，`dfs(v)` 必然會在 `dfs(u)` 之前完成。在逆後序中，$u$ 會在 $v$ 之前。
        -   **情況 3: `dfs(v)` 已經被呼叫但尚未完成。** 這意味著 $v$ 是 $u$ 在 DFS 樹中的一個祖先，並且存在一條從 $v$ 到 $u$ 的路徑。如果同時存在 $u \rightarrow v$ 的邊，那麼圖中就存在一個環 ($u \rightarrow v \leadsto u$)。但我們的前提是 DAG，所以這種情況不會發生。
    -   因此，對於任何邊 $u \rightarrow v$，$u$ 的完成時間總是在 $v$ 的完成時間之後，所以在逆後序中，$u$ 會在 $v$ 之前。
-   **實現:**
    ```java
    boolean[] marked;
    Stack<Integer> reversePostorder; // 用棧來存儲逆後序

    public TopologicalSort(Digraph G) {
        marked = new boolean[G.V()];
        reversePostorder = new Stack<>();
        for (int v = 0; v < G.V(); v++) {
            if (!marked[v]) {
                dfs(G, v);
            }
        }
    }

    private void dfs(Digraph G, int v) {
        marked[v] = true;
        for (int w : G.adj(v)) {
            if (!marked[w]) {
                dfs(G, w);
            }
            // 如果 G.adj(v) 包含已標記但未完成的 w (即 w 在當前遞迴堆疊中)，則檢測到環
            // 但對於拓撲排序，我們假設輸入是 DAG
        }
        reversePostorder.push(v); // v 完成訪問，將其推入棧中
    }

    public Iterable<Integer> getReversePostorder() {
        return reversePostorder; // 棧頂到棧底的順序即為一個拓撲排序
                                 // (或者直接將棧中的元素依次彈出得到拓撲序)
    }
    ```
-   **時間複雜度:** $O(V+E)$，與 DFS 相同。

### 基於入度 (Kahn 演算法) 的拓撲排序演算法

-   **演算法:**
    1.  計算所有頂點的入度。
    2.  初始化一個佇列，將所有入度為 0 的頂點加入佇列。
    3.  初始化一個列表 `topologicalOrder` 用於存儲排序結果。
    4.  當佇列不為空時：
        a.  從佇列中取出一個頂點 `u`，將 `u` 加入 `topologicalOrder`。
        b.  對於 `u` 的每個鄰接點 `v`：
            i.  將 `v` 的入度減 1。
            ii. 如果 `v` 的入度變為 0，則將 `v` 加入佇列。
    5.  如果 `topologicalOrder` 中的頂點數量等於圖中的總頂點數，則排序成功。否則，圖中存在環。
-   **時間複雜度:** $O(V+E)$。

### 注意事項

-   一個 DAG 的拓撲排序可能不唯一。
-   拓撲排序是理解有向無環圖結構和依賴關係的基礎。