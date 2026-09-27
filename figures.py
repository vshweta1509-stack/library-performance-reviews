"""Figures for the manuscript.

Figure 1  importance-performance analysis by library (stated importance).
Figure 2  mean annual star rating by library (supporting figure, not published).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10}
)

SHORT_LABELS = {"AS": "Staff (AS)", "IC": "Access (IC)", "LP": "Place (LP)", "DF": "Depository (DF)"}
# Manual label offsets (points) to stop annotations overlapping.
LABEL_OFFSETS = {
    ("CPL", "AS"): (-40, -36), ("CPL", "DF"): (8, 4), ("CPL", "IC"): (-60, 6), ("CPL", "LP"): (6, 8),
    ("ASM", "AS"): (8, -26), ("ASM", "DF"): (8, -24), ("ASM", "IC"): (8, 4),
    ("DPL", "AS"): (8, -22), ("DPL", "DF"): (-8, -30),
}


def ipa_figure(table: pd.DataFrame, cutoffs: dict, library_names: dict[str, str], path: Path) -> Path:
    """Four-panel IPA chart with bootstrap intervals; open markers mark small samples."""
    libraries = [lib for lib in table["library"].unique() if lib != "ALL"]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True)
    for axis, library in zip(axes.flat, libraries):
        panel = table[table["library"] == library]
        axis.axvline(cutoffs["stated_importance_mean"], color="grey", lw=0.8, ls="--")
        axis.axhline(cutoffs["nss_mean"], color="grey", lw=0.8, ls="--")
        for _, row in panel.iterrows():
            filled = not row["small_sample"]
            axis.errorbar(
                row["stated_importance_pct"], row["nss"],
                yerr=[[row["nss"] - row["nss_ci_low"]], [row["nss_ci_high"] - row["nss"]]],
                fmt="o", color="black", mfc="black" if filled else "white", capsize=3, lw=0.8,
            )
            offset = LABEL_OFFSETS.get((library, row["dimension"]), (6, 4))
            axis.annotate(
                f"{SHORT_LABELS[row['dimension']]}\nn={int(row['n_mentions'])}",
                (row["stated_importance_pct"], row["nss"]), xytext=offset, textcoords="offset points", fontsize=8,
            )
        axis.set_title(library_names[library], fontsize=10)
        axis.set_xlim(0, 70)
        axis.set_ylim(-40, 105)
    for axis in axes[1]:
        axis.set_xlabel("Importance: share of reviews mentioning the dimension (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Performance: Net Sentiment Score")
    figure.text(
        0.5, 0.005,
        "Dashed lines = grand means across library–dimension pairs. Bars = 95% bootstrap CI. Open markers: n < 30.",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=[0, 0.02, 1, 1])
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def temporal_figure(rated: pd.DataFrame, libraries: list[str], library_names: dict[str, str], path: Path,
                    first_year: int = 2016, last_year: int = 2026) -> Path:
    """Mean annual star rating by library."""
    styles = {"NLI": "-o", "CPL": "--s", "ASM": "-.^", "DPL": ":D"}
    yearly = (
        rated[rated["year"].between(first_year, last_year)]
        .groupby(["lib", "year"], observed=True)["rating"].mean().reset_index()
    )
    figure, axis = plt.subplots(figsize=(8, 4.2))
    for library in libraries:
        series = yearly[yearly["lib"] == library]
        axis.plot(series["year"], series["rating"], styles.get(library, "-o"), color="black", ms=4, lw=1,
                  label=library_names[library])
    axis.set_ylim(3.6, 4.8)
    axis.set_xlabel("Year (approximate for libraries with relative dates)")
    axis.set_ylabel("Mean star rating")
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path
