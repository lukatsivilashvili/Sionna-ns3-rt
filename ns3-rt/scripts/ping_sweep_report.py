#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def read_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (vals[0], 0.0)
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return (mu, math.sqrt(var))


def group_by_distance(rows, use_actual):
    grouped = defaultdict(list)
    for row in rows:
        d_actual = to_float(row.get("distance_actual_m"))
        d_param = to_float(row.get("distance_param_m"))
        key = d_actual if use_actual and d_actual is not None else d_param
        if key is None:
            continue
        grouped[key].append(row)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def fmt_num(v, digits=2, sci_digits=3, sci_threshold=1e-4):
    if v is None:
        return "NA"
    try:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
    except Exception:
        pass
    if v != 0 and abs(v) < sci_threshold:
        return f"{v:.{sci_digits}e}"
    return f"{v:.{digits}f}"


def write_summary(summary_path: Path, grouped, mode_label=None, include_mode=False, append=False):
    mode_value = mode_label if mode_label is not None else ""
    open_mode = "a" if append else "w"
    with summary_path.open(open_mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            header = [
                "distance_m",
                "runs",
                "tx_mean",
                "tx_std",
                "rx_mean",
                "rx_std",
                "loss_pct_mean",
                "loss_pct_std",
                "avg_ms_mean",
                "avg_ms_std",
                "pathgain_db_mean",
                "pathgain_db_std",
                "delay_s_mean",
                "delay_s_std",
                "los_mean",
                "los_std",
            ]
            if include_mode:
                header = ["mode"] + header
            writer.writerow(header)

        for d, rows in grouped.items():
            tx_vals = [to_float(r.get("tx")) for r in rows]
            rx_vals = [to_float(r.get("rx")) for r in rows]
            loss_vals = [to_float(r.get("loss_pct")) for r in rows]
            avg_vals = [to_float(r.get("avg_ms")) for r in rows]
            path_vals = [to_float(r.get("sionna_pathgain_db")) for r in rows]
            delay_vals = [to_float(r.get("sionna_delay_s")) for r in rows]
            los_vals = [to_float(r.get("sionna_los")) for r in rows]

            tx_mu, tx_sd = mean_std(tx_vals)
            rx_mu, rx_sd = mean_std(rx_vals)
            loss_mu, loss_sd = mean_std(loss_vals)
            avg_mu, avg_sd = mean_std(avg_vals)
            path_mu, path_sd = mean_std(path_vals)
            delay_mu, delay_sd = mean_std(delay_vals)
            los_mu, los_sd = mean_std(los_vals)

            row = [
                fmt_num(d, digits=2),
                len(rows),
                fmt_num(tx_mu, digits=1),
                fmt_num(tx_sd, digits=1),
                fmt_num(rx_mu, digits=1),
                fmt_num(rx_sd, digits=1),
                fmt_num(loss_mu, digits=1),
                fmt_num(loss_sd, digits=1),
                fmt_num(avg_mu, digits=2),
                fmt_num(avg_sd, digits=2),
                fmt_num(path_mu, digits=2),
                fmt_num(path_sd, digits=2),
                fmt_num(delay_mu, digits=3),
                fmt_num(delay_sd, digits=3),
                fmt_num(los_mu, digits=2),
                fmt_num(los_sd, digits=2),
            ]
            if include_mode:
                row = [mode_value] + row
            writer.writerow(row)


def plot_summary(plot_path: Path, grouped, use_actual):
    plt = try_import_matplotlib()
    if plt is None:
        return False

    distances = list(grouped.keys())
    loss_mean = []
    loss_std = []
    avg_mean = []
    avg_std = []
    path_mean = []
    path_std = []
    delay_mean = []
    delay_std = []

    for d in distances:
        rows = grouped[d]
        tx_mu, tx_sd = mean_std([to_float(r.get("tx")) for r in rows])
        rx_mu, rx_sd = mean_std([to_float(r.get("rx")) for r in rows])
        loss_mu, loss_sd = mean_std([to_float(r.get("loss_pct")) for r in rows])
        avg_mu, avg_sd = mean_std([to_float(r.get("avg_ms")) for r in rows])
        path_mu, path_sd = mean_std([to_float(r.get("sionna_pathgain_db")) for r in rows])
        delay_mu, delay_sd = mean_std([to_float(r.get("sionna_delay_s")) for r in rows])

        def to_nan(v):
            return float("nan") if v is None else v

        loss_mean.append(to_nan(loss_mu))
        loss_std.append(to_nan(loss_sd))
        avg_mean.append(to_nan(avg_mu))
        avg_std.append(to_nan(avg_sd))
        path_mean.append(to_nan(path_mu))
        path_std.append(to_nan(path_sd))
        delay_mean.append(to_nan(delay_mu))
        delay_std.append(to_nan(delay_sd))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    x_label = "Distance (m)" + (" [actual]" if use_actual else " [param]")

    axes[0, 0].errorbar(distances, loss_mean, yerr=loss_std, marker="o", capsize=3)
    axes[0, 0].set_title("Packet Loss (%)")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("Loss %")

    axes[0, 1].errorbar(distances, avg_mean, yerr=avg_std, marker="o", capsize=3)
    axes[0, 1].set_title("Ping RTT (ms)")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("RTT ms")

    axes[1, 0].errorbar(distances, path_mean, yerr=path_std, marker="o", capsize=3)
    axes[1, 0].set_title("Sionna Pathgain (dB)")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Pathgain dB")

    axes[1, 1].errorbar(distances, delay_mean, yerr=delay_std, marker="o", capsize=3)
    axes[1, 1].set_title("Sionna Delay (s)")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Delay s")

    fig.suptitle("Ping + Sionna Metrics vs Distance")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True


def safe_tag(value):
    try:
        tag = f"{float(value):.4g}"
    except Exception:
        tag = str(value)
    return re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_") or "dist"


def plot_per_distance(out_dir: Path, grouped, use_actual, subdir_name="per_distance_plots"):
    plt = try_import_matplotlib()
    if plt is None:
        return 0

    per_dir = out_dir / subdir_name
    per_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for d, rows in grouped.items():
        def run_key(r):
            run_val = to_float(r.get("run"))
            return run_val if run_val is not None else 0.0

        rows_sorted = sorted(rows, key=run_key)
        runs = []
        loss_vals = []
        avg_vals = []
        path_vals = []
        delay_vals = []

        for idx, r in enumerate(rows_sorted, start=1):
            run_val = to_float(r.get("run"))
            runs.append(int(run_val) if run_val is not None else idx)

            def to_nan(v):
                return float("nan") if v is None else v

            loss_vals.append(to_nan(to_float(r.get("loss_pct"))))
            avg_vals.append(to_nan(to_float(r.get("avg_ms"))))
            path_vals.append(to_nan(to_float(r.get("sionna_pathgain_db"))))
            delay_vals.append(to_nan(to_float(r.get("sionna_delay_s"))))

        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        x_label = "Run index"
        dist_label = f"{fmt_num(d, digits=2)} m" + (" [actual]" if use_actual else " [param]")

        def has_data(vals):
            return any(not math.isnan(v) for v in vals)

        if has_data(loss_vals):
            axes[0, 0].plot(runs, loss_vals, marker="o")
        else:
            axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("Packet Loss (%)")
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel("Loss %")

        if has_data(avg_vals):
            axes[0, 1].plot(runs, avg_vals, marker="o")
        else:
            axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Ping RTT (ms)")
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel("RTT ms")

        if has_data(path_vals):
            axes[1, 0].plot(runs, path_vals, marker="o")
        else:
            axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Sionna Pathgain (dB)")
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel("Pathgain dB")

        if has_data(delay_vals):
            axes[1, 1].plot(runs, delay_vals, marker="o")
        else:
            axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Sionna Delay (s)")
        axes[1, 1].set_xlabel(x_label)
        axes[1, 1].set_ylabel("Delay s")

        fig.suptitle(f"Per-Run Metrics at Distance {dist_label}")
        out_path = per_dir / f"ping_sweep_d{safe_tag(d)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1

    return count


def write_markdown_report(report_path: Path, grouped, use_actual):
    lines = []
    lines.append("# Ping Sweep Report")
    lines.append("")
    lines.append(f"- Distances: {len(grouped)}")
    if grouped:
        runs = [len(v) for v in grouped.values()]
        lines.append(f"- Runs per distance: min={min(runs)}, max={max(runs)}")
    lines.append(f"- Distance axis: {'actual' if use_actual else 'param'}")
    lines.append("")
    lines.append("## Summary (mean ± std)")
    lines.append("")
    lines.append("| distance_m | runs | tx_mean | rx_mean | loss_pct_mean | avg_ms_mean | pathgain_db_mean | delay_s_mean | los_mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for d, rows in grouped.items():
        tx_mu, tx_sd = mean_std([to_float(r.get("tx")) for r in rows])
        rx_mu, rx_sd = mean_std([to_float(r.get("rx")) for r in rows])
        loss_mu, loss_sd = mean_std([to_float(r.get("loss_pct")) for r in rows])
        avg_mu, avg_sd = mean_std([to_float(r.get("avg_ms")) for r in rows])
        path_mu, path_sd = mean_std([to_float(r.get("sionna_pathgain_db")) for r in rows])
        delay_mu, delay_sd = mean_std([to_float(r.get("sionna_delay_s")) for r in rows])
        los_mu, los_sd = mean_std([to_float(r.get("sionna_los")) for r in rows])

        def cell(mu, sd):
            if mu is None:
                return "NA"
            if sd is None:
                return fmt_num(mu, digits=2)
            return f"{fmt_num(mu, digits=2)} ± {fmt_num(sd, digits=2)}"

        lines.append(
            f"| {fmt_num(d, digits=2)} | {len(rows)} | {cell(tx_mu, tx_sd)} | {cell(rx_mu, rx_sd)} | "
            f"{cell(loss_mu, loss_sd)} | {cell(avg_mu, avg_sd)} | {cell(path_mu, path_sd)} | "
            f"{cell(delay_mu, delay_sd)} | {cell(los_mu, los_sd)} |"
        )

    report_path.write_text("\n".join(lines))


def infer_mode_from_name(csv_path: Path):
    parts = [p.lower() for p in csv_path.parts]
    if any(p in ("baseline", "legacy") for p in parts):
        return "baseline"
    if any(p in ("sionna", "rt") for p in parts):
        return "sionna"
    name = csv_path.name.lower()
    if "baseline" in name or "legacy" in name:
        return "baseline"
    if "sionna" in name:
        return "sionna"
    return "unknown"


def group_by_mode(rows):
    mode_groups = defaultdict(list)
    for row in rows:
        mode = (row.get("mode") or "").strip()
        if not mode:
            mode = "unknown"
        mode_groups[mode].append(row)
    return dict(mode_groups)


def plot_comparison(plot_path: Path, mode_grouped, use_actual):
    plt = try_import_matplotlib()
    if plt is None:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    x_label = "Distance (m)" + (" [actual]" if use_actual else " [param]")

    for mode, grouped in mode_grouped.items():
        distances = list(grouped.keys())
        loss_mean = []
        loss_std = []
        avg_mean = []
        avg_std = []
        path_mean = []
        path_std = []
        delay_mean = []
        delay_std = []

        for d in distances:
            rows = grouped[d]
            loss_mu, loss_sd = mean_std([to_float(r.get("loss_pct")) for r in rows])
            avg_mu, avg_sd = mean_std([to_float(r.get("avg_ms")) for r in rows])
            path_mu, path_sd = mean_std([to_float(r.get("sionna_pathgain_db")) for r in rows])
            delay_mu, delay_sd = mean_std([to_float(r.get("sionna_delay_s")) for r in rows])

            def to_nan(v):
                return float("nan") if v is None else v

            loss_mean.append(to_nan(loss_mu))
            loss_std.append(to_nan(loss_sd))
            avg_mean.append(to_nan(avg_mu))
            avg_std.append(to_nan(avg_sd))
            path_mean.append(to_nan(path_mu))
            path_std.append(to_nan(path_sd))
            delay_mean.append(to_nan(delay_mu))
            delay_std.append(to_nan(delay_sd))

        def has_data(vals):
            return any(not math.isnan(v) for v in vals)

        if has_data(loss_mean):
            axes[0, 0].errorbar(distances, loss_mean, yerr=loss_std, marker="o", capsize=3, label=mode)
        if has_data(avg_mean):
            axes[0, 1].errorbar(distances, avg_mean, yerr=avg_std, marker="o", capsize=3, label=mode)
        if has_data(path_mean):
            axes[1, 0].errorbar(distances, path_mean, yerr=path_std, marker="o", capsize=3, label=mode)
        if has_data(delay_mean):
            axes[1, 1].errorbar(distances, delay_mean, yerr=delay_std, marker="o", capsize=3, label=mode)

    axes[0, 0].set_title("Packet Loss (%)")
    axes[0, 0].set_xlabel(x_label)
    axes[0, 0].set_ylabel("Loss %")
    axes[0, 0].legend()

    axes[0, 1].set_title("Ping RTT (ms)")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel("RTT ms")
    axes[0, 1].legend()

    axes[1, 0].set_title("Sionna Pathgain (dB)")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel("Pathgain dB")
    axes[1, 0].legend()

    axes[1, 1].set_title("Sionna Delay (s)")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel("Delay s")
    axes[1, 1].legend()

    fig.suptitle("Ping + Sionna Metrics vs Distance (Combined)")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True


def write_markdown_report_combined(report_path: Path, mode_grouped, use_actual):
    lines = []
    lines.append("# Ping Sweep Report (Combined)")
    lines.append("")
    lines.append(f"- Modes: {', '.join(sorted(mode_grouped.keys()))}")
    lines.append(f"- Distance axis: {'actual' if use_actual else 'param'}")
    lines.append("")

    for mode in sorted(mode_grouped.keys()):
        grouped = mode_grouped[mode]
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append(f"- Distances: {len(grouped)}")
        if grouped:
            runs = [len(v) for v in grouped.values()]
            lines.append(f"- Runs per distance: min={min(runs)}, max={max(runs)}")
        lines.append("")
        lines.append("### Summary (mean ± std)")
        lines.append("")
        lines.append(
            "| distance_m | runs | tx_mean | rx_mean | loss_pct_mean | avg_ms_mean | pathgain_db_mean | delay_s_mean | los_mean |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for d, rows in grouped.items():
            tx_mu, tx_sd = mean_std([to_float(r.get("tx")) for r in rows])
            rx_mu, rx_sd = mean_std([to_float(r.get("rx")) for r in rows])
            loss_mu, loss_sd = mean_std([to_float(r.get("loss_pct")) for r in rows])
            avg_mu, avg_sd = mean_std([to_float(r.get("avg_ms")) for r in rows])
            path_mu, path_sd = mean_std([to_float(r.get("sionna_pathgain_db")) for r in rows])
            delay_mu, delay_sd = mean_std([to_float(r.get("sionna_delay_s")) for r in rows])
            los_mu, los_sd = mean_std([to_float(r.get("sionna_los")) for r in rows])

            def cell(mu, sd):
                if mu is None:
                    return "NA"
                if sd is None:
                    return fmt_num(mu, digits=2)
                return f"{fmt_num(mu, digits=2)} ± {fmt_num(sd, digits=2)}"

            lines.append(
                f"| {fmt_num(d, digits=2)} | {len(rows)} | {cell(tx_mu, tx_sd)} | {cell(rx_mu, rx_sd)} | "
                f"{cell(loss_mu, loss_sd)} | {cell(avg_mu, avg_sd)} | {cell(path_mu, path_sd)} | "
                f"{cell(delay_mu, delay_sd)} | {cell(los_mu, los_sd)} |"
            )

        lines.append("")

    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot ping sweep CSV")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to ping_sweep.csv (can be repeated)",
    )
    parser.add_argument("--out-dir", default="results", help="Output directory")
    parser.add_argument("--use-actual-distance", action="store_true", help="Use distance_actual_m if present")
    parser.add_argument(
        "--per-distance-plots",
        action="store_true",
        help="Generate one plot per distance value (requires matplotlib)",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mode_out_dirs = {}
    for csv_path in csv_paths:
        csv_rows = read_rows(csv_path)
        inferred_mode = infer_mode_from_name(csv_path)
        mode_out_dirs.setdefault(inferred_mode, csv_path.parent)
        for row in csv_rows:
            if not (row.get("mode") or "").strip():
                row["mode"] = inferred_mode
            mode_out_dirs.setdefault(row["mode"], csv_path.parent)
            rows.append(row)
    use_actual = args.use_actual_distance
    mode_groups = group_by_mode(rows)

    if len(mode_groups) == 1:
        grouped = group_by_distance(rows, use_actual)

        summary_path = out_dir / "ping_sweep_summary.csv"
        write_summary(summary_path, grouped)

        plot_path = out_dir / "ping_sweep_plots.png"
        plotted = plot_summary(plot_path, grouped, use_actual)

        report_path = out_dir / "ping_sweep_report.md"
        write_markdown_report(report_path, grouped, use_actual)

        print(f"Wrote {summary_path}")
        print(f"Wrote {report_path}")
        if plotted:
            print(f"Wrote {plot_path}")
        else:
            print("matplotlib not available; skipped plot generation.")
        if args.per_distance_plots:
            per_count = plot_per_distance(out_dir, grouped, use_actual)
            if per_count:
                print(f"Wrote {per_count} per-distance plots to {out_dir / 'per_distance_plots'}")
            else:
                print("matplotlib not available; skipped per-distance plots.")
        return

    # Multi-mode combine
    combined_summary_path = out_dir / "ping_sweep_summary.csv"
    append = False
    mode_grouped = {}

    for mode, mode_rows in sorted(mode_groups.items()):
        grouped = group_by_distance(mode_rows, use_actual)
        mode_grouped[mode] = grouped

        mode_tag = safe_tag(mode)
        mode_dir = mode_out_dirs.get(mode, out_dir / mode_tag)
        mode_dir.mkdir(parents=True, exist_ok=True)

        mode_summary_path = mode_dir / "ping_sweep_summary.csv"
        write_summary(mode_summary_path, grouped)
        write_summary(combined_summary_path, grouped, mode_label=mode, include_mode=True, append=append)
        append = True

        plot_path = mode_dir / "ping_sweep_plots.png"
        plotted = plot_summary(plot_path, grouped, use_actual)
        if plotted:
            print(f"Wrote {plot_path}")
        else:
            print("matplotlib not available; skipped plot generation.")

        if args.per_distance_plots:
            per_count = plot_per_distance(mode_dir, grouped, use_actual, subdir_name="per_distance_plots")
            if per_count:
                print(f"Wrote {per_count} per-distance plots to {mode_dir / 'per_distance_plots'}")
            else:
                print("matplotlib not available; skipped per-distance plots.")

    combined_plot_path = out_dir / "ping_sweep_plots.png"
    combined_plotted = plot_comparison(combined_plot_path, mode_grouped, use_actual)

    report_path = out_dir / "ping_sweep_report.md"
    write_markdown_report_combined(report_path, mode_grouped, use_actual)

    print(f"Wrote {combined_summary_path}")
    print(f"Wrote {report_path}")
    if combined_plotted:
        print(f"Wrote {combined_plot_path}")
    else:
        print("matplotlib not available; skipped combined plot generation.")


if __name__ == "__main__":
    main()
