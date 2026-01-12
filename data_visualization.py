import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("preprocessed_data.csv", index_col="timestamp", parse_dates=True)

print("Descriptive statistical results：")
print(data[["moisture_value", "normalized_value"]].describe())

plt.figure(figsize=(12, 6))
plt.plot(data.index, data["normalized_value"], label="Normalized Moisture Value")
plt.title("Soil Moisture Trend Over 48 Hours")
plt.xlabel("Time")
plt.ylabel("Normalized Moisture Value (0=Wet, 1=Dry)")
plt.legend()
plt.grid(True)
plt.savefig("moisture_trend.png", dpi=300, bbox_inches="tight")
plt.show()
