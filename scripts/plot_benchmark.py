"""Render the recorded retrieval quality/latency tradeoff without rerunning inference."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "reports/scifact-v1/report.json").read_text())
names = list(report["summary"])
rows = list(report["summary"].values())
x = np.arange(len(names))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
values = np.array([r["ndcg_at_10"] for r in rows])
intervals = np.array([r["ndcg_ci95"] for r in rows])
axes[0].bar(x, values, color=["#79919f", "#527f98", "#147a78", "#9db9b5"])
axes[0].errorbar(
    x,
    values,
    yerr=[values - intervals[:, 0], intervals[:, 1] - values],
    fmt="none",
    color="#263b4b",
    capsize=4,
)
axes[0].set(ylabel="nDCG@10", ylim=(0, 1), title="Relevance · 300 queries, bootstrap 95% intervals")
axes[1].bar(x, [r["p50_ms"] for r in rows], color="#147a78")
axes[1].set(
    ylabel="Milliseconds (log scale)", yscale="log", title="Measured warmed query latency · median"
)
for ax in axes:
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
fig.suptitle("SciFact: complete 5,183-document corpus", fontsize=15)
for suffix in ["png", "svg"]:
    fig.savefig(root / f"reports/scifact-study.{suffix}", dpi=180)
