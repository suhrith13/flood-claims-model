"""Step 5: combine the results into one table and draw three charts.

Reads  : outputs/naive_metrics.json, honest_metrics.json, importance_*.csv,
         honest_test_predictions.parquet
Writes : outputs/results.csv, results.md, and three PNG charts
"""
import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUTPUT_DIR


def results_table():
    naive = json.load(open(OUTPUT_DIR / "naive_metrics.json"))
    honest = json.load(open(OUTPUT_DIR / "honest_metrics.json"))
    runs = honest["runs"]  # median baseline, policy-math baseline, honest random, honest time
    rows = runs[:2] + [naive] + runs[2:]
    df = pd.DataFrame(rows)[["setup", "features", "split", "mae_ratio", "mae_dollars", "r2"]]
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    lines = ["| Setup | Features | Split | MAE (ratio) | MAE ($ per claim) | R2 |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.setup} | {r.features} | {r.split} | {r.mae_ratio:.3f} | "
                     f"{r.mae_dollars:,.0f} | {r.r2:.3f} |")
    clf = honest["classifier"]
    lines.append("")
    lines.append(f"'Paid at all' classifier (honest features, time split): AUC {clf['auc']:.3f}; "
                 f"{clf['share_paid_in_test']:.1%} of test claims were paid something.")
    (OUTPUT_DIR / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


def chart_importance():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name, title in [(axes[0], "naive", "Naive model (leaky)"),
                            (axes[1], "honest", "Honest model")]:
        imp = pd.read_csv(OUTPUT_DIR / f"importance_{name}.csv").head(10).iloc[::-1]
        ax.barh(imp["feature"], imp["share"])
        ax.set_title(title)
        ax.set_xlabel("share of total importance (gain)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importance_naive_vs_honest.png", dpi=130)
    plt.close()


def chart_calibration(preds):
    preds = preds.copy()
    preds["bin"] = pd.qcut(preds["predicted"], 10, duplicates="drop")
    cal = preds.groupby("bin", observed=True)[["predicted", "paid_ratio"]].mean()
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.plot(cal["predicted"], cal["paid_ratio"], "o-", label="honest model")
    plt.xlabel("average predicted ratio (per decile)")
    plt.ylabel("average actual ratio")
    plt.title("Calibration on future claims")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration.png", dpi=130)
    plt.close()


def chart_error_by_state(preds):
    preds = preds.copy()
    preds["abs_err"] = (preds["predicted"] - preds["paid_ratio"]).abs()
    by_state = preds.groupby("state")["abs_err"].agg(["mean", "size"])
    top = by_state.sort_values("size", ascending=False).head(15).sort_values("mean")
    plt.figure(figsize=(7, 5.5))
    plt.barh(top.index, top["mean"])
    plt.xlabel("mean absolute error (ratio)")
    plt.title("Error by state (15 states with most test claims)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_by_state.png", dpi=130)
    plt.close()


def main():
    results_table()
    preds = pd.read_parquet(OUTPUT_DIR / "honest_test_predictions.parquet")
    chart_importance()
    chart_calibration(preds)
    chart_error_by_state(preds)
    print("Saved charts to outputs/")


if __name__ == "__main__":
    main()
