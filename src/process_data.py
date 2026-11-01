import os

with open("data/raw_data.csv", "r") as f:
    lines = f.readlines()[1:]

os.makedirs("output", exist_ok=True)
with open("output/metrics.txt", "w") as f:
    f.write(f"sample_count: {len(lines)}\n")

print(f"Metrics saved: {len(lines)} samples.")