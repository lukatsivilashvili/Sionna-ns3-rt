#!/usr/bin/env python3
import glob
import xml.etree.ElementTree as ET
import re
import csv
import os

FLOWMON_DIR = "results/flowmon"
OUT_CSV = "results/csv/results.csv"
DEST_PORT = "9000"

os.makedirs("results/csv", exist_ok=True)

def to_seconds(v):
    if v.endswith("ns"): return float(v[:-2]) * 1e-9
    if v.endswith("us"): return float(v[:-2]) * 1e-6
    if v.endswith("ms"): return float(v[:-2]) * 1e-3
    if v.endswith("s"):  return float(v[:-1])
    return 0.0

aggregates = {}

for xml_path in glob.glob(f"{FLOWMON_DIR}/*.xml"):
    name = os.path.basename(xml_path)

    # Extract model + distance from filename
    m = re.search(r"(baseline|sionna).*?d(\d+)", name)
    if not m:
        continue

    model = m.group(1)
    distance = int(m.group(2))

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Build flowId -> destinationPort map ---
    flow_ports = {}
    for flow in root.findall(".//Flow"):
        fid = flow.get("flowId")
        dport = flow.get("destinationPort")
        if fid and dport:
            flow_ports[fid] = dport

    # --- Parse statistics ---
    for fs in root.findall(".//FlowStats/Flow"):
        fid = fs.get("flowId")
        if flow_ports.get(fid) != DEST_PORT:
            continue

        tx_packets = int(fs.get("txPackets", "0"))
        rx_packets = int(fs.get("rxPackets", "0"))
        rx_bytes   = int(fs.get("rxBytes", "0"))

        t1 = to_seconds(fs.get("timeFirstRxPacket", "0ns"))
        t2 = to_seconds(fs.get("timeLastRxPacket", "0ns"))
        duration = max(t2 - t1, 1e-9)

        throughput_mbps = (rx_bytes * 8) / duration / 1e6 if rx_bytes > 0 else 0.0
        loss_pct = ((tx_packets - rx_packets) / tx_packets * 100) if tx_packets > 0 else 0.0

        # Truncate to hundredths (round down).
        throughput_mbps = int(throughput_mbps * 100) / 100
        loss_pct = int(loss_pct * 100) / 100

        key = (distance, model)
        if key not in aggregates:
            aggregates[key] = {
                "throughput_sum": 0.0,
                "loss_sum": 0.0,
                "tx_packets_sum": 0,
                "rx_packets_sum": 0,
                "count": 0,
            }
        agg = aggregates[key]
        agg["throughput_sum"] += throughput_mbps
        agg["loss_sum"] += loss_pct
        agg["tx_packets_sum"] += tx_packets
        agg["rx_packets_sum"] += rx_packets
        agg["count"] += 1

rows = []
for (distance, model), agg in aggregates.items():
    count = agg["count"]
    if count == 0:
        continue
    avg_throughput = agg["throughput_sum"] / count
    avg_loss = agg["loss_sum"] / count
    avg_tx_packets = agg["tx_packets_sum"] / count
    avg_rx_packets = agg["rx_packets_sum"] / count

    # Truncate to hundredths (round down).
    avg_throughput = int(avg_throughput * 100) / 100
    avg_loss = int(avg_loss * 100) / 100

    rows.append([
        distance,
        model,
        avg_throughput,
        avg_loss,
        int(avg_tx_packets),
        int(avg_rx_packets)
    ])

rows.sort(key=lambda r: (r[0], r[1]))

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "distance_m",
        "model",
        "throughput_mbps",
        "loss_pct",
        "tx_packets",
        "rx_packets"
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT_CSV}")
