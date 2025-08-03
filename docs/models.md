# 🤖 Models

This section describes the deep learning models implemented to forecast appliance energy consumption based on historical multivariate time series data.

---

## 1️⃣ LSTM (Long Short-Term Memory)

LSTMs are a type of recurrent neural network capable of learning long-term dependencies.  
They are well-suited for time series forecasting tasks due to their memory cell structure.

**Architecture Highlights:**
- Input: Sliding windows of 50 time steps
- Layers: 2 LSTM layers + Dropout + Dense
- Loss: MSE  
- Optimizer: Adam

**Strengths:**
- Captures temporal patterns
- Robust to moderate noise

---

## 2️⃣ TCN (Temporal Convolutional Network)

TCNs use 1D dilated convolutions with residual connections to model long-range dependencies efficiently.

**Architecture Highlights:**
- Input: Same window size
- Multiple convolutional blocks with increasing dilation
- Residual + ReLU + Dropout layers
- Fully connected output layer

**Advantages:**
- Fast to train
- Handles long sequences without recurrent loops

---

## 3️⃣ Transformer

Originally designed for NLP, Transformers can handle long-range dependencies using self-attention.

**Architecture Highlights:**
- Positional encoding for temporal information
- Multi-head self-attention layers
- Feed-forward dense layers
- Output linear projection

**Pros:**
- High parallelization
- Good for multivariate time series
- Robust to noise and variable importance

---

## 📈 Training Strategy

- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metrics:** MAE and RMSE
- **Train/Test Split:** Last 100 periods for testing
- **Early Stopping:** Based on validation loss
- **Hardware:** GPU-accelerated (if available)

---

## 📊 Results Summary

| Model        | MAE     | RMSE    |
|--------------|---------|---------|
| LSTM         | 0.472   | 0.889   |
| Transformer  | 0.456   | 0.882   |
| TCN          | 0.537   | 0.905   |

---

Each model was trained using the same data splits and preprocessing pipeline for fair comparison.