import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("preprocessed_data.csv", index_col="timestamp", parse_dates=True)
values = data["normalized_value"].values.reshape(-1, 1)

look_back = 24
X, y = [], []
for i in range(look_back, len(values)):
    X.append(values[i-look_back:i, 0])
    y.append(values[i, 0])
X, y = np.array(X), np.array(y)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Linear regression model MSE：{mse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Moisture Value")
plt.plot(y_pred, label="Predicted Moisture Value", linestyle="--")
plt.title("Linear Regression: Actual vs Predicted Moisture (12-Hour Forecast)")
plt.xlabel("Time Steps (30 Minutes Each)")
plt.ylabel("Normalized Moisture Value")
plt.legend()
plt.savefig("linear_regression_result.png", dpi=300, bbox_inches="tight")
plt.show()
