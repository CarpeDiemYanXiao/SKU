"""
批量测试脚本
依次在5个100k测试集上运行 train.py，汇总最终指标
"""

import os
import sys
import re
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

DATASETS = [
    "100k_sku.parquet",
    "100k_sku_v2.parquet",
    "100k_sku_v3.parquet",
    "100k_sku_v4.parquet",
    "100k_sku_v6.parquet",
]

CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
LOG_DIR = Path(__file__).parent / "output"


def write_report(log_file, timestamp, results):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Batch Test Report  {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Dataset':<25} {'ACC':>8} {'RTS':>8} {'Status':>8}\n")
        f.write("-" * 55 + "\n")
        for r in results:
            f.write(f"{r['dataset']:<25} {r['acc']:>7.2f}% {r['rts']:>7.2f}% {r['status']:>8}\n")
        f.write("-" * 55 + "\n")

        if results:
            avg_acc = sum(r["acc"] for r in results) / len(results)
            avg_rts = sum(r["rts"] for r in results) / len(results)
            all_pass = all(r["status"] == "PASS" for r in results)
        else:
            avg_acc, avg_rts, all_pass = 0.0, 0.0, False

        f.write(f"{'Average':<25} {avg_acc:>7.2f}% {avg_rts:>7.2f}%\n\n")
        f.write(f"Verdict: {'ALL PASS' if all_pass else 'NOT ALL PASS'}\n")
        f.write(f"Target: ACC >= 80%, RTS <= 2.4%\n")

    return avg_acc, avg_rts


def update_config(dataset_name):
    rel_path = f"../data/{dataset_name}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["train_path"] = rel_path
    cfg["data"]["eval_path"] = rel_path
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_best_metrics(output_text):
    acc, rts = None, None
    for line in output_text.splitlines():
        m = re.search(r"Best ACC:\s*([\d.]+)%.*Best RTS:\s*([\d.]+)%", line)
        if m:
            acc, rts = float(m.group(1)), float(m.group(2))
    if acc is None:
        for line in reversed(output_text.splitlines()):
            m = re.search(r"Best:\s*ACC=([\d.]+)%\s*RTS=([\d.]+)%", line)
            if m:
                acc, rts = float(m.group(1)), float(m.group(2))
                break
    return acc, rts


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"batch_test_{timestamp}.txt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    work_dir = str(Path(__file__).parent)

    results = []
    print("=" * 60)
    print(f"Batch test: {len(DATASETS)} datasets")
    print("=" * 60)

    for i, ds in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] {ds}")
        update_config(ds)

        proc = subprocess.run(
            [python, "train.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        acc, rts = parse_best_metrics(output)

        if acc is not None:
            status = "PASS" if acc >= 80.0 and rts <= 2.4 else "FAIL"
            print(f"    ACC={acc:.2f}%  RTS={rts:.2f}%  [{status}]")
        else:
            acc, rts, status = 0.0, 0.0, "ERROR"
            print(f"    Failed to parse metrics")

        results.append({"dataset": ds, "acc": acc, "rts": rts, "status": status})
        avg_acc, avg_rts = write_report(log_file, timestamp, results)

    # 终端输出汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<25} {'ACC':>8} {'RTS':>8} {'Status':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['dataset']:<25} {r['acc']:>7.2f}% {r['rts']:>7.2f}% {r['status']:>8}")
    print("-" * 55)
    print(f"{'Average':<25} {avg_acc:>7.2f}% {avg_rts:>7.2f}%")
    print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
