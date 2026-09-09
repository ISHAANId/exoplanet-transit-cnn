"""
Step 8: compare Experiment A (raw/cleaned, no phase folding) vs Experiment B
(cleaned + phase folded), both fed through the identical CNN architecture and
evaluated on the identical held-out test stars.

Run after both experiments have been trained:
    python scripts/05_train_experiment.py --variant raw
    python scripts/05_train_experiment.py --variant folded
    python scripts/06_compare_experiments.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR  # noqa: E402


def main():
    path_a = RESULTS_DIR / "experiment_A_raw" / "metrics.json"
    path_b = RESULTS_DIR / "experiment_B_folded" / "metrics.json"

    if not path_a.exists() or not path_b.exists():
        print("Missing results -- run 05_train_experiment.py for both --variant raw and "
              "--variant folded first.")
        return

    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        rows.append({
            "metric": metric,
            "A_raw_no_fold": a[metric],
            "B_phase_folded": b[metric],
            "delta_B_minus_A": (b[metric] - a[metric]) if (a[metric] is not None and
                                                             b[metric] is not None) else None,
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    out_path = RESULTS_DIR / "experiment_comparison.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved comparison table -> {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = table["metric"].tolist()
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width / 2, table["A_raw_no_fold"], width, label="A: raw (no fold)")
        ax.bar(x + width / 2, table["B_phase_folded"], width, label="B: phase folded")
        ax.set_xticks(x, metrics, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Experiment A vs B on held-out test stars")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "experiment_comparison.png", dpi=150)
        print(f"Saved comparison chart -> {RESULTS_DIR / 'experiment_comparison.png'}")
    except ImportError:
        pass

    verdict = "B (phase folding) outperformed A" if b["accuracy"] > a["accuracy"] else (
        "A (no folding) outperformed B" if a["accuracy"] > b["accuracy"] else "A and B tied")
    print(f"\nOn accuracy: {verdict}.")
    print("Remember this verdict is only as reliable as the test set is large -- check n_test "
          "in both metrics.json files before drawing conclusions.")


if __name__ == "__main__":
    main()
