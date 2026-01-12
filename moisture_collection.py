import csv
import time
from datetime import datetime
from gpiozero import DigitalInputDevice

sensor = DigitalInputDevice(17)
csv_path = "moisture_data.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "moisture_status", "moisture_value"])

total_hours = 48
interval = 30 * 60
iterations = total_hours * 2

for i in range(iterations):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    value = 1 if sensor.is_active else 0
    status = "Dry" if value == 1 else "Wet"
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, status, value])
    
    print(f"Collection completed{i+1}：{now} - {status}")
    time.sleep(interval)
