import pandas as pd
import numpy as np

data = pd.read_csv("moisture_data.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data.set_index("timestamp", inplace=True)

data["denoised_value"] = data["moisture_value"].rolling(window=3).mean()
data = data.dropna()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data["normalized_value"] = scaler.fit_transform(data[["denoised_value"]])

data.to_csv("preprocessed_data.csv")
print("Data preprocessing is completed and saved as preprocessed_data.csv")
print("The first 5 rows of the preprocessed data：")
print(data.head())
