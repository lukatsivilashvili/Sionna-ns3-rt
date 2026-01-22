import csv
from collections import defaultdict
import matplotlib.pyplot as plt

aggregates = defaultdict(lambda: defaultdict(list))

with open("results/csv/results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = int(row["distance_m"])
        model = row["model"]
        thr = float(row["throughput_mbps"])
        loss = float(row["loss_pct"])
        aggregates[model][d].append((thr, loss))

data = defaultdict(list)
for model, per_distance in aggregates.items():
    for d, samples in per_distance.items():
        if not samples:
            continue
        avg_thr = sum(s[0] for s in samples) / len(samples)
        avg_loss = sum(s[1] for s in samples) / len(samples)
        data[model].append((d, avg_thr, avg_loss))

for model in data:
    data[model].sort(key=lambda x: x[0])

# Throughput plot
plt.figure()
for model, pts in data.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, marker="o", label=model)

plt.xlabel("Distance (m)")
plt.ylabel("Throughput (Mbps)")
plt.title("LTE Throughput vs Distance (Baseline vs Sionna RT)")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/throughput_vs_distance.png", dpi=200)

# Loss plot
plt.figure()
for model, pts in data.items():
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    plt.plot(xs, ys, marker="o", label=model)

plt.xlabel("Distance (m)")
plt.ylabel("Packet loss (%)")
plt.title("LTE Packet Loss vs Distance (Baseline vs Sionna RT)")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/loss_vs_distance.png", dpi=200)

print("✔ Wrote plots to results/plots/")
