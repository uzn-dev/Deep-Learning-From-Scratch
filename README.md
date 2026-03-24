# Deep Learning From Scratch

這裡將會記錄我從零開始學習深度學習的過程。

## 專案目錄

* [**`01_neuron_basics/`**](./01_neuron_basics/)：神經網路基礎與前向傳播 (Forward Propagation)
  * [`01_single_neuron/`](./01_neuron_basics/01_single_neuron/) - 單一神經元的數學運算 ($w \cdot x + b$) 與 ReLU 激勵函數。
  * [`02_mnist_forward/`](./01_neuron_basics/02_mnist_forward/) - 導入 NumPy 矩陣運算，實作多層網路的 MNIST 影像前向推論與 Softmax 機率輸出。

* [**`02_micrograd_engine/`**](./02_micrograd_engine/)：自動微分引擎與反向傳播 (Backpropagation) 🚀 *(進行中)*
  * [`01_engine_impl/`](./02_micrograd_engine/01_engine_impl/) - 從純量 (Scalar) 開始，從零打造支援自動微分的 `Value` 計算節點。