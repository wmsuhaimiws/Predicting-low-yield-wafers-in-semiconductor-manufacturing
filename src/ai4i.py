"""
ai4i.py — optional IIoT predictive-maintenance mini-study (creativity add-on).

Secondary to SECOM. Uses UCI AI4I 2020 (id=601, 10,000 x 14, 5 failure modes).
Where SECOM asks "which signals predict a bad wafer?", this asks "which machine
conditions predict an impending failure?" — the same pattern for equipment uptime.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_ai4i

_FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # tool-wear, heat, power, overstrain, random


def prepare_ai4i(df: pd.DataFrame):
    """Engineer model-ready features and the binary 'Machine failure' target."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Air temperature [K]": "air_temp_K",
        "Process temperature [K]": "proc_temp_K",
        "Rotational speed [rpm]": "rot_speed_rpm",
        "Torque [Nm]": "torque_Nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if {"proc_temp_K", "air_temp_K"}.issubset(df.columns):
        df["temp_diff_K"] = df["proc_temp_K"] - df["air_temp_K"]
    if {"torque_Nm", "rot_speed_rpm"}.issubset(df.columns):
        df["power_W"] = df["torque_Nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60.0

    qcol = next((c for c in df.columns if c.lower() in ("type", "product type")), None)
    if qcol:
        df = pd.get_dummies(df, columns=[qcol], prefix="type", drop_first=True)

    feature_cols = [c for c in [
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "temp_diff_K", "power_W",
    ] if c in df.columns]
    feature_cols += [c for c in df.columns if c.startswith("type_")]

    target = "machine_failure" if "machine_failure" in df.columns else None
    return df, feature_cols, target


def run_ai4i_study(save_fig: bool = True):
    """Load -> prepare -> model -> figure. Returns (metrics dict, figure path)."""
    raw = load_ai4i()
    df, feats, target = prepare_ai4i(raw)
    if target is None:
        raise ValueError("AI4I: could not locate the 'Machine failure' target column")

    X, y = df[feats], df[target].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=config.RANDOM_STATE)
    sc = StandardScaler().fit(X_tr)
    rf.fit(sc.transform(X_tr), y_tr)
    proba = rf.predict_proba(sc.transform(X_te))[:, 1]
    auc = roc_auc_score(y_te, proba)
    report = classification_report(y_te, (proba > 0.5).astype(int),
                                   target_names=["healthy", "failure"], zero_division=0)

    importance = (pd.Series(rf.feature_importances_, index=feats)
                  .sort_values(ascending=False))

    fig_path = None
    if save_fig:
        present_modes = [m for m in _FAILURE_MODES if m in df.columns]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if present_modes:
            counts = df[present_modes].sum().sort_values(ascending=False)
            axes[0].bar(counts.index, counts.values, color=config.COLOR_FAIL)
            axes[0].set_title("AI4I failure modes (count of events)")
            axes[0].set_ylabel("events")
        importance.head(8)[::-1].plot.barh(ax=axes[1], color=config.COLOR_ACCENT)
        axes[1].set_title(f"Drivers of machine failure (ROC-AUC={auc:.2f})")
        fig.tight_layout()
        fig_path = config.FIGURES_DIR / "11_ai4i_failure_modes.png"
        fig.savefig(fig_path, dpi=config.FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[ai4i] wrote {fig_path.name}")

    return {"roc_auc": auc, "report": report,
            "top_drivers": importance.head(8).to_dict()}, fig_path


if __name__ == "__main__":  # pragma: no cover
    metrics, _ = run_ai4i_study()
    print(metrics["report"])
    print("ROC-AUC:", round(metrics["roc_auc"], 3))
