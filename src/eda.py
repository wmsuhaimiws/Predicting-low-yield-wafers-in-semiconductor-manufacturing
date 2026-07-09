"""
eda.py — exploratory data analysis figures.

Generates 8 publication-quality figures into figures/. Each function applies
Few / Evergreen principles: the right chart for the data, a high data-ink ratio,
muted gridlines and a single reserved red accent for the *fail* class.

Run standalone:  python -m src.eda
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from . import config

# Two-colour map for boolean missingness masks (light = present, red = missing)
_MISS_CMAP = ListedColormap(["#ECF0F1", config.COLOR_FAIL])

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": config.FIG_DPI,
    "savefig.dpi": config.FIG_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.size": 11,
})

_PAL = {0: config.COLOR_PASS, 1: config.COLOR_FAIL}
_LBL = {0: "pass", 1: "fail"}


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {out.name}")
    return out


def plot_class_balance(y: pd.Series):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([_LBL[i] for i in counts.index], counts.values,
                  color=[_PAL[i] for i in counts.index])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/total:.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("number of wafers")
    ax.set_title(f"Severe class imbalance (~{counts.max()//counts.min()}:1)")
    ax.margins(y=0.18)
    return _save(fig, "01_class_balance.png")


def plot_missingness(X_raw: pd.DataFrame):
    miss = X_raw.isna().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sample_cols = miss.head(60).index
    sns.heatmap(X_raw[sample_cols].isna().T, cbar=False, cmap=_MISS_CMAP, ax=ax)
    ax.set_title("Missingness map — worst 60 sensors (red = missing)")
    ax.set_xlabel("wafer (record) index")
    ax.set_yticks([])
    f2 = _save(fig, "02_missingness_map.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(miss.values * 100, bins=40, color=config.COLOR_ACCENT, edgecolor="white")
    ax.axvline(config.MISSING_COL_THRESHOLD * 100, color=config.COLOR_FAIL,
               ls="--", lw=2, label=f"drop threshold ({config.MISSING_COL_THRESHOLD:.0%})")
    ax.set_xlabel("% missing within a sensor")
    ax.set_ylabel("number of sensors")
    ax.set_title("Most sensors are nearly complete; a tail is mostly empty")
    ax.legend()
    f3 = _save(fig, "03_missingness_hist.png")
    return f2, f3


def plot_variance_distribution(X_imputed: pd.DataFrame):
    var = X_imputed.var(numeric_only=True).replace(0, np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(var.values), bins=40, color=config.COLOR_ACCENT, edgecolor="white")
    ax.set_xlabel("log10(variance)")
    ax.set_ylabel("number of sensors")
    ax.set_title("Many sensors carry little variance (left tail ≈ near-constant)")
    return _save(fig, "04_variance_distribution.png")


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, top_sensors):
    cols = list(top_sensors)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), cols):
        for cls in (0, 1):
            sns.kdeplot(X.loc[y == cls, col], ax=ax, fill=True, alpha=0.35,
                        color=_PAL[cls], label=_LBL[cls], warn_singular=False)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    fig.suptitle("Top sensors: pass vs fail distribution shift", fontweight="bold")
    return _save(fig, "05_feature_distributions.png")


def plot_correlation_heatmap(X: pd.DataFrame, top_sensors):
    cols = list(top_sensors)[:20]
    corr = X[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="vlag", center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Correlation among top sensors (redundancy check)")
    return _save(fig, "06_correlation_heatmap.png")


def plot_feature_importance(imp_top: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp_top["sensor"][::-1], imp_top["importance"][::-1],
            color=config.COLOR_ACCENT)
    ax.set_xlabel("Random-Forest importance")
    ax.set_title("Top predictive sensors = engineering leads")
    return _save(fig, "07_top_feature_importance.png")


def plot_passfail_boxplots(X: pd.DataFrame, y: pd.Series, top_sensors):
    cols = list(top_sensors)[:6]
    long = X[cols].copy()
    long[config.LABEL_COL] = y.map(_LBL).values
    long = long.melt(id_vars=config.LABEL_COL, var_name="sensor", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=long, x="sensor", y="value", hue=config.LABEL_COL,
                palette={"pass": config.COLOR_PASS, "fail": config.COLOR_FAIL},
                fliersize=1, ax=ax)
    ax.set_title("Where top sensors separate pass from fail")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "08_passfail_boxplots.png")


def run_all(X_raw, X_clean, y, top_sensors, imp_top):
    """Convenience: regenerate every EDA figure in one call."""
    plot_class_balance(y)
    plot_missingness(X_raw)
    plot_variance_distribution(X_clean)
    plot_feature_distributions(X_clean, y, top_sensors)
    plot_correlation_heatmap(X_clean, top_sensors)
    plot_feature_importance(imp_top)
    plot_passfail_boxplots(X_clean, y, top_sensors)
