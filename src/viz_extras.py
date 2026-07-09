"""
viz_extras.py — creativity & innovation extras.

1. yield_pareto()   A Pareto view: do a small minority of sensors drive most of
                    the model's predictive signal? (The "vital few".)
2. build_dashboard() A self-contained interactive Plotly dashboard exported to
                    docs/interactive_dashboard.html.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config


def yield_pareto(imp_full: pd.DataFrame, top_n: int = 25, save: bool = True):
    """Pareto chart of feature importance: bars = importance, line = cumulative %."""
    d = imp_full.head(top_n).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d["sensor"], y=d["importance"], name="importance",
                marker_color=config.COLOR_ACCENT)
    fig.add_scatter(x=d["sensor"], y=d["cum_importance"] * 100, name="cumulative %",
                    mode="lines+markers", line=dict(color=config.COLOR_FAIL, width=3),
                    secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="grey", secondary_y=True,
                  annotation_text="80%")
    fig.update_layout(
        title="Yield-signal Pareto — the vital few sensors",
        template="plotly_white", height=520,
        xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1),
    )
    fig.update_yaxes(title_text="RF importance", secondary_y=False)
    fig.update_yaxes(title_text="cumulative %", range=[0, 105], secondary_y=True)
    if save:
        out = config.FIGURES_DIR / "09_yield_pareto.png"
        try:
            fig.write_image(str(out))     # needs kaleido
            print(f"[viz_extras] wrote {out.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[viz_extras] PNG export skipped ({exc}); HTML still produced")
        fig.write_html(str(config.DOCS_DIR / "yield_pareto.html"))
    return fig


def build_dashboard(X_clean: pd.DataFrame, y: pd.Series,
                    imp_full: pd.DataFrame, results_df: pd.DataFrame,
                    out_path: Path | None = None) -> Path:
    """Compose a 2x2 interactive dashboard -> docs/interactive_dashboard.html."""
    out_path = out_path or (config.DOCS_DIR / "interactive_dashboard.html")
    top = imp_full.head(15)
    counts = y.value_counts().sort_index()
    best_sensor = imp_full.iloc[0]["sensor"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Class balance (pass vs fail)", "Top 15 predictive sensors",
            f"Most predictive sensor: {best_sensor}", "Model comparison (PR-AUC)",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}]],
        vertical_spacing=0.14, horizontal_spacing=0.12,
    )
    fig.add_bar(x=["pass", "fail"], y=counts.values,
                marker_color=[config.COLOR_PASS, config.COLOR_FAIL],
                showlegend=False, row=1, col=1)
    fig.add_bar(y=top["sensor"][::-1], x=top["importance"][::-1], orientation="h",
                marker_color=config.COLOR_ACCENT, showlegend=False, row=1, col=2)
    for cls, name, col in [(0, "pass", config.COLOR_PASS), (1, "fail", config.COLOR_FAIL)]:
        fig.add_box(y=X_clean.loc[y == cls, best_sensor], name=name,
                    marker_color=col, boxpoints="outliers", row=2, col=1)
    if results_df is not None and len(results_df):
        fig.add_bar(x=results_df["model"], y=results_df["PR-AUC"],
                    marker_color=config.COLOR_ACCENT, showlegend=False, row=2, col=2)
    fig.update_layout(
        title_text="SECOM Yield Optimization — interactive dashboard",
        template="plotly_white", height=820, width=1100, margin=dict(t=90),
    )
    fig.update_xaxes(tickangle=-30, row=2, col=2)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"[viz_extras] wrote {out_path}")
    return out_path
