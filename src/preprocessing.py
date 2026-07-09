"""
preprocessing.py — data cleaning & preparation.

Each step returns a *before/after summary* so the notebook can print evidence
that the operation did what we claim. The written justification for every step
lives in docs/methodology.md; the one-line rationale is repeated inline here.

Pipeline order (leakage-safe):
    encode_label -> drop_high_missing -> median_impute
    -> drop_near_constant -> winsorize_outliers
    (scaling + SMOTE are applied INSIDE the modelling Pipeline, train-fold only.)
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningReport:
    """Audit trail of what cleaning did — printed as the before/after summary."""
    n_rows: int = 0
    n_cols_start: int = 0
    n_cols_after_missing: int = 0
    n_cols_after_variance: int = 0
    missing_cells_start: int = 0
    missing_cells_after_impute: int = 0
    dropped_high_missing: list[str] = field(default_factory=list)
    dropped_near_constant: list[str] = field(default_factory=list)
    class_balance_raw: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "stage": [
                "rows", "columns (start)", "columns (after missing-drop)",
                "columns (after variance-drop)", "missing cells (start)",
                "missing cells (after impute)", "sensors dropped: >50% missing",
                "sensors dropped: near-constant",
            ],
            "value": [
                self.n_rows, self.n_cols_start, self.n_cols_after_missing,
                self.n_cols_after_variance, self.missing_cells_start,
                self.missing_cells_after_impute, len(self.dropped_high_missing),
                len(self.dropped_near_constant),
            ],
        })


def encode_label(y_raw: pd.Series) -> pd.Series:
    """Map SECOM's -1/+1 labels to 0/1 with fail = 1 (the minority event)."""
    mapping = {config.LABEL_PASS_RAW: 0, config.LABEL_FAIL_RAW: 1}
    y = y_raw.map(mapping).astype(int)
    y.name = config.LABEL_COL
    return y


def drop_high_missing(X: pd.DataFrame, threshold: float = config.MISSING_COL_THRESHOLD):
    """Drop sensors whose missing fraction exceeds `threshold` (can't impute >half honestly)."""
    miss_frac = X.isna().mean()
    to_drop = miss_frac[miss_frac > threshold].index.tolist()
    return X.drop(columns=to_drop), to_drop


def median_impute(X: pd.DataFrame):
    """Fill remaining NaNs with each column's median (robust to skew/outliers)."""
    medians = X.median(numeric_only=True)
    return X.fillna(medians), medians


def drop_near_constant(X: pd.DataFrame,
                       dominant_threshold: float = config.DOMINANT_VALUE_THRESHOLD):
    """Drop constant / near-constant sensors (>99% one value) — no discriminative power."""
    to_drop: list[str] = []
    for col in X.columns:
        s = X[col]
        if s.nunique(dropna=False) <= 1:
            to_drop.append(col)
            continue
        dominant_share = s.value_counts(normalize=True, dropna=False).iloc[0]
        if dominant_share > dominant_threshold:
            to_drop.append(col)
    return X.drop(columns=to_drop), to_drop


def winsorize_outliers(X: pd.DataFrame, limits: tuple[float, float] = config.WINSOR_LIMITS):
    """Clip each sensor to [1st, 99th] percentile — tame glitches, keep every wafer."""
    lo = X.quantile(limits[0])
    hi = X.quantile(limits[1])
    return X.clip(lower=lo, upper=hi, axis=1)


def clean_secom(X_raw: pd.DataFrame, y_raw: pd.Series, winsorize: bool = True):
    """Run the full cleaning sequence and return (X_clean, y, report). Not scaled (see modeling)."""
    report = CleaningReport(
        n_rows=len(X_raw),
        n_cols_start=X_raw.shape[1],
        missing_cells_start=int(X_raw.isna().sum().sum()),
        class_balance_raw=y_raw.value_counts().to_dict(),
    )
    y = encode_label(y_raw)

    X, dropped_missing = drop_high_missing(X_raw)
    report.n_cols_after_missing = X.shape[1]
    report.dropped_high_missing = dropped_missing

    X, _medians = median_impute(X)
    report.missing_cells_after_impute = int(X.isna().sum().sum())

    X, dropped_const = drop_near_constant(X)
    report.n_cols_after_variance = X.shape[1]
    report.dropped_near_constant = dropped_const

    if winsorize:
        X = winsorize_outliers(X)

    return X.reset_index(drop=True), y.reset_index(drop=True), report


def imbalance_ratio(y: pd.Series) -> float:
    """Majority:minority ratio — the headline imbalance number (~14:1)."""
    counts = y.value_counts()
    return counts.max() / counts.min()
