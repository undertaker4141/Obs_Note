# Shell 程式設計 (Introductory Bourne Shell Programming)

(來源: c12_shell_programming.pdf)

## Shell Script 簡介
-   Shell Script 是一個由 Shell 指令組成的程式，儲存在一個普通的 UNIX 檔案中，並在 Shell 環境中逐行執行。
-   Shell Script 支援控制流程指令，類似高階程式語言 (如 C)，可以實現：
    -   非循序執行 (例如，使用 `if`, `case`)
    -   重複執行指令區塊 (例如，使用 `while`, `for`)

## 執行 Bourne Shell Script
有三種主要方法執行 Bourne Shell Script (假設腳本檔名為 `script_file`)：

1.  **使其可執行並直接執行**：
    ```bash
    chmod u+x script_file
    ./script_file
    ```
    如果目前目錄 (`.`) 在您的 `PATH` 環境變數中，可以直接用 `script_file` 執行。

2.  **使用明確的 Shell 執行**：
    ```bash
    /bin/sh script_file
    ```
    如果 `/bin` 在 `PATH` 中，可以簡化為 `sh script_file`。

3.  **使用 Shebang (#! /bin/sh)**：
    在腳本檔案的第一行加入 `#!/bin/sh`。
    ```bash
    #!/bin/sh
    # 腳本的其他內容...
    chmod u+x script_file
    ./script_file
    ```
    當目前的 Shell 遇到 `#!`，它會將該行餘下的部分視為要執行的 Shell 的絕對路徑，並用該 Shell 來執行此腳本。這是最常用的方法，可以確保腳本在指定的 Shell 環境下執行，無論使用者目前的 Shell 是什麼。

## Shell 變數與相關指令

### Shell 變數是什麼？
-   變數是用於儲存資料的具名記憶體位置。可以透過名稱而非位址來參考它。

### Shell 變數的類型
-   **環境變數 (Environment Variables)**：例如 `PATH`, `HOME`。
    -   用於自訂 Shell 環境，並會被子行程繼承。
    -   大多數環境變數在登入時由系統級的 `/etc/profile` 初始化。
    -   使用者可以在自己的 `~/.profile` 檔案中進一步設定，例如：`export PATH="$PATH:/my/custom/bin"`。
-   **使用者定義變數 (User-defined Variables)**：例如 `myvar=42`。
    -   主要用於 Shell Script 中的暫存。

### 重要可寫入 Bourne Shell 環境變數 (部分範例)
-   `CDPATH`: `cd` 指令的搜尋路徑。
-   `EDITOR`: 預設編輯器名稱。
-   `HOME`: 使用者家目錄的路徑。
-   `MAIL`: 使用者系統信箱檔案的路徑。
-   `PATH`: Shell 搜尋外部指令的路徑序列。
-   `PS1`: 主要 Shell 提示符 (通常是 `$` 或 `#`)。
-   `PS2`: 次要 Shell 提示符 (通常是 `>`)，用於多行指令。
-   `PWD`: 目前工作目錄的絕對路徑。
-   `TERM`: 使用者終端機類型。

### 重要唯讀 Bourne Shell 環境變數 (部分範例)
-   `$0`: 腳本或函式名稱。
-   `$1` - `$9`: 命令列參數 1 到 9。
-   `$*`: 所有命令列參數，視為一個單一字串。
-   `$@`: 所有命令列參數，若用雙引號包圍 (`"$@"`)，則每個參數被視為獨立的字串。
-   `$#`: 命令列參數的總數。
-   `$$`: 目前 Shell 行程的 PID。
-   `$?`: 最近執行指令的結束狀態 (0 代表成功，非 0 代表失敗)。
-   `$!`: 最近在背景執行的行程的 PID。

### 使用者定義變數
-   主要用於 Shell Script 中的暫存。
-   不需事先宣告，未初始化的變數預設為空字串 (null string)。
-   `set` 指令 (不加參數) 可顯示所有 Shell 變數及其值。

### 讀取與寫入 Shell 變數
-   **賦值語法**: `variable1=value1 [variable2=value2 ...]`
    -   等號 (`=`) 前後 **不可有空格**。
    -   若值包含空格，需用引號包圍。
    -   單引號 (`'`)：保留所有字元的字面意義 (除了單引號本身)。
    -   雙引號 (`"`)：保留大多數字元的字面意義，但允許變數擴展 (`$var`)、指令替換 (`` `command` `` 或 `$(command)`) 和跳脫字元 (`\`)。
    -   反斜線 (`\`)：保留其後字元的字面意義 (除了換行符 `\n`)。
-   **存取變數值**: 在變數名前加 `$`，例如 `echo $name`。
-   **指令替換 (Command Substitution)**:
    -   語法：`` `command` `` 或 `$(command)`
    -   執行 `command` 並將其標準輸出替換到該位置。
    -   範例：`current_date=$(date)` 或 `files_in_dir=`ls``

### 匯出環境 (`export`)
-   在 Shell 中建立的變數，其子 Shell 預設無法存取。
-   `export` 指令將變數的值傳遞給後續執行的 Shell (子 Shell)。
    -   語法：`export [name-list]`
    -   範例：`name="John Doe"; export name` 或 `export name="John Doe"`
-   傳遞的是變數值的 **副本**。子 Shell 修改該變數不會影響父 Shell 中的原始變數。

### 重設變數 (`unset`)
-   將變數重設為 null (預設初始值) 或移除變數。
    -   語法：`unset [name-list]`
    -   範例：`unset myvar`
-   也可透過賦予空值來重設：`country=`

### 從標準輸入讀取 (`read`)
-   用於互動式 Shell Script，提示使用者輸入並將輸入儲存到變數。
-   語法：`read variable-list`
    -   讀取標準輸入的一行，按空白 (IFS 中的字元) 分割成單字，依序賦值給 `variable-list` 中的變數。
    -   若單字數多於變數數，最後一個變數獲得剩餘所有單字。
    -   若單字數少於變數數，多餘的變數被設為 null。
-   `echo -n "提示訊息: "`：`-n` 選項使游標停在同一行，方便輸入。

### 傳遞參數給 Shell Script
-   命令列參數依序儲存在位置參數 (positional parameters) `$1`, `$2`, `$3`, ... 中。
-   `$0` 儲存腳本本身的名稱。
-   `$#` 儲存傳遞的參數總數。
-   `$*` 和 `$@` 儲存所有參數。
    -   `"$*"`: 所有參數視為一個字串 (例如 `"arg1 arg2 arg3"`)。
    -   `"$@"`: 每個參數視為獨立的引號字串 (例如 `"arg1" "arg2" "arg3"`)，常用於迴圈中處理參數。
-   **`shift [N]` 指令**:
    -   將命令列參數向左移動 N 個位置 (預設為 1)。`$1` 被移出，`$2` 變 `$1`，以此類推。
    -   用於處理超過 9 個參數或逐個處理參數。
-   **`set` 指令**:
    -   `set -- argument-list`: 用 `argument-list` 中的值重設位置參數。`--` 用於防止 `argument-list` 中以 `-` 開頭的參數被 `set` 誤認為選項。
    -   常與指令替換結合使用，例如 `set -- $(date)` 將 `date` 指令的輸出分割並設為位置參數。

## 程式控制流程指令

### `if-then-elif-else-fi` 陳述式
-   用於實現雙向或多向分支。
-   **基本語法 (雙向)**:
    ```bash
    if expression
    then
        then-command-list
    fi
    ```
    或
    ```bash
    if expression
    then
        then-command-list
    else
        else-command-list
    fi
    ```
-   **多向分支語法**:
    ```bash
    if expression1
    then
        then-command-list1
    elif expression2
    then
        elif-command-list1
    elif expression3
    then
        elif-command-list2
    ...
    else
        else-command-list
    fi
    ```
-   `expression` 通常使用 `test` 指令或 `[[ expression ]]` (Bash/Ksh 功能更強) 進行評估。
    -   `test expression` 或 `[ expression ]` (注意 `[` 後和 `]` 前的空格)。
    -   **檔案測試運算子**: `-d file` (目錄), `-f file` (普通檔案), `-r file` (可讀), `-w file` (可寫), `-x file` (可執行), `-s file` (大小非零)。
    -   **整數比較運算子**: `int1 -eq int2` (等於), `-ne` (不等於), `-gt` (大於), `-ge` (大於等於), `-lt` (小於), `-le` (小於等於)。
    -   **字串比較運算子**: `str1 = str2` (相等), `str1 != str2` (不相等), `-n str` (長度非零), `-z str` (長度為零)。
    -   **邏輯運算子**: `! expr` (NOT), `expr1 -a expr2` (AND), `expr1 -o expr2` (OR)。在 `[[...]]` 中使用 `&&` 和 `||`。
-   `exit N` 指令用於終止腳本並返回結束狀態 `N` 給父行程。`0` 通常表示成功，非 `0` 表示錯誤。

### `for` 陳述式 (迴圈)
-   用於重複執行指令區塊，遍歷一個列表。
-   **語法**:
    ```bash
    for variable [in argument-list]
    do
        command-list
    done
    ```
    -   若省略 `in argument-list`，則迴圈遍歷目前的位置參數 (`$@`)。
    -   `argument-list` 可以是空格分隔的字串、變數擴展、指令替換或檔案萬用字元擴展。

### `while` 陳述式 (迴圈)
-   當 `expression` 為真 (結束狀態為 0) 時，重複執行 `command-list`。
-   **語法**:
    ```bash
    while expression
    do
        command-list
    done
    ```
-   常用於無限迴圈 (例如伺服器程式) 或直到滿足某條件為止。
-   若要停止前景無限迴圈，可按 `<Ctrl+C>`；若忽略，可按 `<Ctrl+Z>` 暫停，再用 `kill -9 PID` 強制終止。

### `until` 陳述式 (迴圈)
-   當 `expression` 為假 (結束狀態非 0) 時，重複執行 `command-list`。
-   **語法**:
    ```bash
    until expression
    do
        command-list
    done
    ```
-   語義上與 `while ! expression` 相同。

### `break` 和 `continue` 指令
-   用於改變迴圈的正常執行流程。
-   `break [n]`: 跳出目前 (或第 n 層) 的 `for`, `while`, `until` 迴圈。
-   `continue [n]`: 跳過目前迴圈的剩餘指令，開始下一次迭代 (或跳到第 n 層迴圈的下一次迭代)。
-   通常與 `if` 陳述式結合使用。

### `case` 陳述式
-   提供類似巢狀 `if` 的多向分支機制，但通常更易讀。適用於測試單一變數是否匹配多個模式。
-   **語法**:
    ```bash
    case test-string in
        pattern1)
            command-list1
            ;;
        pattern2|pattern3) # 多個模式用 | 分隔
            command-list2
            ;;
        patternN)
            command-listN
            ;;
        *) # 預設模式，匹配任何其他情況
            default-command-list
            ;;
    esac
    ```
    -   `test-string` 會與每個 `pattern` 進行比較。
    -   `pattern` 可以包含 Shell 萬用字元 (如 `*`, `?`, `[]`)。
    -   第一個匹配的 `pattern` 對應的 `command-list` 會被執行。
    -   每個指令區塊必須以 `;;` 結束。
    -   `*)` 是可選的，用於處理所有其他不匹配的情況。

## 指令群組
-   可以將多個 Shell 指令作為一個群組來執行。
-   **語法**:
    1.  `(command-list)`: `command-list` 在一個子 Shell (subshell) 中執行。子 Shell 中的變數賦值、目錄變更等不會影響目前的 Shell 環境。
    2.  `{ command-list; }`: `command-list` 在目前的 Shell 環境中執行。注意 `{` 和 `command-list` 之間以及 `command-list` 和 `}` 之間的空格，以及最後一個指令後的 `;` (或換行)。
-   指令群組的標準輸出和標準錯誤可以像單個指令一樣被重導向。
    -   範例：`(date; pwd) > output.txt` 或 `{ date; pwd; } > output.txt`

---
[[Unix學習筆記_總覽]]
