# 01_neuron_basics: 神經網路基礎實作

這個章節是從零開始建構深度學習模型的第一步。在此專案中，我完全不依賴高階的深度學習框架（如 PyTorch 或 TensorFlow 的模型建構 API），僅使用純 Python 與 NumPy，手刻出神經網路的底層數學邏輯。

## 📂 資料夾結構與專案說明

### 1. `01_single_neuron` 
* **檔案:** `neuron_from_scratch.py`
* **實作內容:** * 建構了最基礎的神經元數學模型：$z = w \cdot x + b$。
  * 理解權重（Weights）與偏差（Bias）的物理意義。
  * 實作基礎的激勵函數（Activation Function）並觀察其對輸出的非線性影響。

### 2. `02_mnist_forward`
* **檔案:** `mnist_forward_prop.py`
* **實作內容:** * **真實數據處理:** 載入 MNIST 手寫數字資料集，實作影像資料的拉直（Flattening，$28 \times 28 \rightarrow 784$）與數值歸一化（Normalization）。
  * **矩陣化運算:** 從單一神經元的 `for` 迴圈，升級為處理 $(Batch, Features)$ 維度的 NumPy 矩陣乘法（`X @ W + b`），大幅提升運算效率。
  * **權重初始化:** 使用常態分佈 `np.random.randn()` 初始化權重矩陣，並理解為何不能將權重全部設為 0（打破對稱性 Symmetry Breaking）。
  * **進階激勵函數:** * 實作 **ReLU** 函數處理隱藏層的非線性特徵。
    * 實作 **Softmax** 函數將輸出層的分數轉換為機率分佈，並加入了減去最大值的處理以確保**數值穩定性（Numerical Stability）**，防止指數運算溢位（Overflow）。

## 🧠 核心學習重點 (Takeaways)

1. **維度的掌控 (Dimensionality):** 在 Python/NumPy 中，理解一維陣列 `(n,)` 與二維行向量 `(n, 1)` 的差異至關重要。在實作 Softmax 時，深刻體會了 `keepdims=True` 對於廣播機制（Broadcasting）和維度對齊的重要性。
2. **前向傳播 (Forward Pass):** 成功讓真實的影像數據流過隨機初始化的兩層網路，並輸出 10 個類別的預測機率，完成神經網路的「推論」骨架。