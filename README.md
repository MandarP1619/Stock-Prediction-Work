# 📈 Stock Prediction Work

This project implements a stock price prediction system using **LSTM (Long Short-Term Memory)** neural networks. It downloads historical stock data, preprocesses it, trains an LSTM model for each stock, and visualizes the predicted vs. actual opening prices.

---

## 🚀 Features

- 📥 **Automatic Data Collection** via Yahoo Finance (`yfinance`)
- 🔄 **Data Normalization** using MinMaxScaler
- 🧠 **LSTM Model** built with TensorFlow/Keras
- 📊 **Visualizations** of actual vs. predicted stock prices
- 📈 **Evaluation** using Mean Squared Error (MSE)
- 🔁 Predicts multiple stocks: `AAPL`, `AMZN`, `GOOGL`, `NFLX`, `META`

---

## 🧾 File Structure

```
├── .gitignore         # Ignored files/folders
├── LICENSE.md         # License for this project
├── README.md          # You're reading it!
└── script.py          # Core script for stock prediction
```

---

## 🛠️ How It Works

### 🔹 Step-by-Step Pipeline

1. **Download Data**: Retrieves historical `Open` prices using `yfinance`.
2. **Preprocess**: Scales the data and creates sequences using a sliding window.
3. **Model**: Builds an LSTM-based model using TensorFlow/Keras.
4. **Train/Test Split**: 80% training, 20% testing.
5. **Predict & Evaluate**: Makes predictions and computes Mean Squared Error (MSE).
6. **Visualize**: Displays full-range and zoomed-in plots of predictions.

---

## 🧪 Sample Output

For each stock symbol, the model prints:

```
AMZN - Mean Squared Error: 2.35
GOOGL - Mean Squared Error: 1.94
...
```

And generates visual plots:
- 🔴 Actual Opening Price
- 🔵 Predicted Opening Price

---

## 📦 Requirements

Install dependencies via pip:

```bash
pip install numpy pandas yfinance scikit-learn matplotlib tensorflow
```

---

## ▶️ How to Run

```bash
python script.py
```

Make sure you're connected to the internet to fetch the stock data.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE.md` file for more details.
