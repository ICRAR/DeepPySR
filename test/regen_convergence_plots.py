"""
Regenerate convergence plots for Feynman (I.13.4, I.9.18) and BMI longitudinal
from existing CSVs. Optimised for 3-up subfigure layout in the paper:
- Small figure size (fits 0.31 textwidth at 200 dpi)
- Larger fonts for readability at reduced size
- No params text box (clutter at small size)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FEYNMAN_DIR = os.path.join(os.path.dirname(__file__), "feynman/convergence_results")
BMI_DIR     = os.path.join(os.path.dirname(__file__), "bmi/convergence_results/longitudinal")
PAPER_DIR   = os.path.join(os.path.dirname(__file__), "../paper/figures")

PLOTS = [
    {
        "csv":      os.path.join(FEYNMAN_DIR, "eq_I_13_4/convergence_Feynman: I.13.4.csv"),
        "out":      os.path.join(PAPER_DIR, "convergence_I13_4.png"),
        "title":    "Feynman I.13.4",
    },
    {
        "csv":      os.path.join(FEYNMAN_DIR, "eq_I_9_18/convergence_Feynman: I.9.18.csv"),
        "out":      os.path.join(PAPER_DIR, "convergence_I9_18.png"),
        "title":    "Feynman I.9.18",
    },
    {
        "csv":      os.path.join(BMI_DIR, "convergence_Longitudinal.csv"),
        "out":      os.path.join(PAPER_DIR, "convergence_bmi_longitudinal.png"),
        "title":    "RAINE BMI Longitudinal",
    },
]

MODEL_LABELS = {"deeppysr": "DeepPySR", "pysr": "PySR"}
MODEL_COLORS = {"deeppysr": "#1976D2", "pysr": "#E65100"}

FIG_W, FIG_H = 3.2, 2.4   # inches — scales well to 0.31 textwidth
FONTSIZE_TITLE  = 10
FONTSIZE_AXIS   = 8
FONTSIZE_TICK   = 8
FONTSIZE_LEGEND = 8

for cfg in PLOTS:
    df = pd.read_csv(cfg["csv"])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for model in df["Model"].unique():
        mdf = df[df["Model"] == model].sort_values("Iteration")
        label = MODEL_LABELS.get(model, model.upper())
        color = MODEL_COLORS.get(model, None)
        ax.plot(mdf["Iteration"], mdf["Loss"], label=label, color=color, linewidth=1.4)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Iteration", fontsize=FONTSIZE_AXIS)
    ax.set_ylabel("Loss (MSE)", fontsize=FONTSIZE_AXIS)
    ax.set_title(cfg["title"], fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=cfg.get("ytick_size", FONTSIZE_TICK))
    ax.grid(True, which="major", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.legend(fontsize=FONTSIZE_LEGEND, loc="upper right", framealpha=0.8)

    fig.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    fig.savefig(cfg["out"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cfg['out']}")