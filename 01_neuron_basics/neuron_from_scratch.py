import numpy as np

# ==========================================
# 任務一：實作線性轉換 (Linear Transformation)
# ==========================================
def linear_forward(x, w, b):
    """
    計算神經元的線性組合: z = w * x + b
    
    參數:
    x (numpy array): 輸入的特徵向量 (例如長度為 3 的一維陣列)
    w (numpy array): 權重向量 (必須和 x 長度一樣)
    b (float): 偏差值
    
    回傳:
    z (float): 線性計算後的結果
    """
    # 在 Python 裡 NumPy 提供了算內積的函數：np.dot(陣列A, 陣列B)。
    
    z = np.dot(x, w) + b
    return z

# ==========================================
# 任務二：實作激勵函數 (Activation Functions)
# ==========================================
def sigmoid(z):
    """
    實作 Sigmoid 函數: 1 / (1 + e^(-z))
    """
    # 自然對數 e 的次方，可以使用 np.exp()
    
    a = 1/(1+np.exp(-z))
    return a

def relu(z):
    """
    實作 ReLU 函數: max(0, z)
    """
    # 可以使用 np.maximum() 來比較 0 和 z 的大小
    
    a = np.maximum(0,z)
    return a

# ==========================================
# 任務三：整合測試 (Forward Pass)
# ==========================================
if __name__ == "__main__":
    # 1. 準備數據 (假設神經元接收 3 個輸入值)
    x = np.array([1.5, 2.0, -1.0])
    w = np.array([0.5, -0.2, 0.1])
    b = -0.3

    print("輸入 x:", x)
    print("權重 w:", w)
    print("偏差 b:", b)
    print("-" * 30)

    # 2. 開始計算！
    # <--- 呼叫 linear_forward，把 x, w, b 傳進去，並把結果存進變數 z
    z = linear_forward(x, w, b) 
    print(f"線性組合結果 (z): {z}") 
    # 線性組合結果 (z): -0.05000000000000002
    # 因為電腦底層是用二進位在儲存浮點數 (Floating Point)，在轉換過程中產生的極微小誤差。

    # <--- 呼叫 sigmoid，把 z 傳進去
    a_sig = sigmoid(z) 
    print(f"Sigmoid 輸出: {a_sig}")
    # Sigmoid 輸出: 0.4875026035157896

    # <--- 呼叫 relu，把 z 傳進去
    a_relu = relu(z) 
    print(f"ReLU 輸出: {a_relu}")
    # ReLU 輸出: 0.0