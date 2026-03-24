import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    # 1. 下載數據集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print(f"原始圖片形狀: {train_images.shape}") # 應該是 (60000, 28, 28)

    # 2. 挑選一張圖片來看看
    sample_idx = 0
    plt.imshow(train_images[sample_idx], cmap='gray')
    plt.title(f"Label: {train_labels[sample_idx]}")
    plt.show()

    # 3. 數據預處理 (Data Preprocessing)
    # reshape(60000, 784)：將 (28, 28) 的圖片拉直成 (784,) 的向量
    #　.astype('float32')：原本圖片像素是用uint8儲存的，但做數學運算時需要精確的浮點數。
    # / 255：將數值從 0~255 縮放到 0~1 (歸一化)，這對模型訓練非常有幫助
    x_train = train_images.reshape(60000, 784).astype('float32') / 255
    
    print(f"拉直後的圖片形狀: {x_train.shape}") # 應該是 (60000, 784)
    return x_train, train_labels

# ==========================================
# 任務一：初始化權重 (Initialization)
# ==========================================
def init_parameters():
    """
    任務：設定正確的矩陣維度。
    有 784 個輸入像素、128 個隱藏神經元、10 個輸出類別。
    """
    # 1. 第一層：輸入層 -> 隱藏層
    # 提示：形狀應該是 (輸入數量, 隱藏層數量)
    W1 = np.random.randn(784, 128) * 0.01 
    # randn 產生的數字會落在 -3 ~ 3 之間
    # *0.01可以讓初始權重變得很微小，讓模型剛開始學習時比較穩定
    b1 = np.zeros((1, 128)) # 偏差通常是一個列向量：(1, 隱藏層數量)
    
    # 2. 第二層：隱藏層 -> 輸出層
    W2 = np.random.randn(128, 10) * 0.01
    b2 = np.zeros((1, 10))
    
    return W1, b1, W2, b2

# ==========================================
# 任務二：激勵函數 (Activation Functions)
# ==========================================
def relu(Z):
    """
    任務：實作 ReLU 函數 (如果 Z < 0 則為 0，否則為 Z)
    """
    # 使用 NumPy 內建的函數來處理整個矩陣的比大小
    return np.maximum(0,Z)

def softmax(Z):
    """
    任務：實作 Softmax 函數，把分數轉成機率
    公式：e^Z / sum(e^Z)
    """
    # 先減去最大值，防止 C++ 常見的 overflow 問題
    # 最大值會變成0 (e^0 = 1)，其餘值會變負數(e^負數 < 1)，所有數字都在0~1間
    # axis=0 是直向（由上往下），axis=1 是橫向（由左往右）。
    # keepdims=True 保持維度的層級（Rank）不變
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    
    # 請計算 e 的 Z_stable 次方
    expZ = np.exp(Z_stable) 
    
    # 請算出機率分佈 A (將 expZ 除以它自己的總和)
    # 提示：計算總和時要加上參數 axis=1, keepdims=True，確保是橫向相加
    sum_expZ = np.sum(expZ, axis=1, keepdims=True)
    A = expZ / sum_expZ 
    return A

# ==========================================
# 任務三：前向傳播 (Forward Propagation)
# ==========================================
def forward_pass(X, W1, b1, W2, b2):
    """
    任務：把矩陣乘法和激勵函數串起來！
    """
    # 第一層：計算 Z1 (線性) 然後通過 A1 (ReLU)
    # 提示：矩陣乘法可以使用 np.dot(陣列1, 陣列2) 或 Python 的 @ 符號
    Z1 = X @ W1 + b1
    A1 = relu(Z1) 
    
    # 第二層：計算 Z2 (線性) 然後通過 A2 (Softmax)
    Z2 = A1 @ W2 + b2 
    A2 = softmax(Z2)
    
    return A2

# ==========================================
# 整合測試
# ==========================================
if __name__ == "__main__":
    x_train, y_train = load_and_preprocess_data()
    print("-" * 30)
    
    W1, b1, W2, b2 = init_parameters()
    print("權重初始化完成！")
    
    X_sample = x_train[0:1]
    predictions = forward_pass(X_sample, W1, b1, W2, b2)
    
    print("-" * 30)
    print(f"模型的預測機率分佈:\n", np.round(predictions, 3))
    
    predicted_digit = np.argmax(predictions)
    print(f"\n模型猜測這個數字是: {predicted_digit}")
    print(f"實際上的數字是: {y_train[0]}")