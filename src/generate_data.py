import os
import random

os.makedirs("data", exist_ok=True)

with open("data/raw_data.csv", "w") as f:
    f.write("feature1,feature2,target\n")
    for _ in range(100):
        f1 = round(random.uniform(0, 10), 2)
        f2 = round(random.uniform(0, 10), 2)
        target = int(f1 + f2 > 10)
        f.write(f"{f1},{f2},{target}\n")

print("Raw data generated.")