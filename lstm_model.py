import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("preprocessed_data.csv", index_col="timestamp", parse_dates=True)
values = data["normalized_value"].values.reshape(-1, 1)

look_back = 24
X, y = [], []
for i in range(look_back, len(values)):
    X.append(values[i-look_back:i, 0])
    y.append(values[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"LSTM model MSE：{mse:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.title("LSTM Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.savefig("lstm_loss.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Moisture Value")
plt.plot(y_pred, label="Predicted Moisture Value", linestyle="--")
plt.title("LSTM: Actual vs Predicted Moisture (Periodic Forecast)")
plt.xlabel("Time Steps (30 Minutes Each)")
plt.ylabel("Normalized Moisture Value")
plt.legend()
plt.savefig("lstm_result.png", dpi=300, bbox_inches="tight")
plt.show()
