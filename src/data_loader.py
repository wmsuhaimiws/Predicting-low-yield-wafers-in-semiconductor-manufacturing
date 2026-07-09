"""
data_loader.py — robust, reproducible data acquisition.

Design goal: *no hidden local-file dependencies*. Every loader has at least two
fallbacks so the notebook runs end-to-end in a fresh Colab runtime, on a laptop,
or in CI.

Public functions
----------------
load_secom()  -> (features: DataFrame, target: Series, timestamp: Series)
load_ai4i()   -> DataFrame
"""
from __future__ import annotations
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Classic UCI flat-file endpoints (stable for >15 years)
_SECOM_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
_SECOM_LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once; reuse the cached copy on later runs."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[data_loader] downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted UCI host)
    return dest


# --------------------------------------------------------------------------- #
# SECOM (primary dataset, id=179)
# --------------------------------------------------------------------------- #
def _load_secom_via_ucimlrepo():
    """Preferred path: programmatic fetch, no manual download."""
    from ucimlrepo import fetch_ucirepo  # lazy import so the dep stays optional

    secom = fetch_ucirepo(id=179)
    # ucimlrepo occasionally returns features/targets as None (transient API/parse
    # issue). Detect it explicitly and raise so load_secom() uses the flat files.
    if secom.data.features is None or secom.data.targets is None:
        raise ValueError("ucimlrepo returned no features/targets for id=179")
    X = secom.data.features.copy()
    y = secom.data.targets.copy()
    # targets may be a 1-col DataFrame; squeeze to a Series
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    # ucimlrepo exposes the timestamp via the original data when available
    ts = None
    for cand in ("Time", "time", "timestamp"):
        original = getattr(secom.data, "original", pd.DataFrame())
        if cand in original.columns:
            ts = original[cand]
            break
    return X, y, ts


def _load_secom_via_flatfiles():
    """Fallback: parse the original space-delimited UCI flat files."""
    data_path = _download(_SECOM_DATA_URL, config.DATA_RAW / "secom.data")
    label_path = _download(_SECOM_LABELS_URL, config.DATA_RAW / "secom_labels.data")

    # 590 sensor columns, whitespace-separated, 'NaN' tokens for missing values
    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    # labels file: "<label> <YYYY-MM-DD HH:MM:SS>"  e.g. "-1 2008-07-19 11:55:00"
    raw = pd.read_csv(label_path, sep=r"\s+", header=None,
                      names=["label", "date", "clock"])
    y = raw["label"].astype(int)
    ts = pd.to_datetime(raw["date"] + " " + raw["clock"], errors="coerce")
    return X, y, ts


def load_secom():
    """
    Return (features, target, timestamp) for the SECOM dataset.

    target is the *raw* label (-1 pass / +1 fail); encoding to 0/1 happens in
    preprocessing.encode_label so the raw signal is preserved end-to-end.
    """
    try:
        X, y, ts = _load_secom_via_ucimlrepo()
        print(f"[data_loader] SECOM via ucimlrepo: {X.shape[0]} rows x {X.shape[1]} sensors")
    except Exception as exc:  # noqa: BLE001 — fall back on *any* failure
        print(f"[data_loader] ucimlrepo unavailable ({exc!r}); using flat-file fallback")
        X, y, ts = _load_secom_via_flatfiles()
        print(f"[data_loader] SECOM via flat files: {X.shape[0]} rows x {X.shape[1]} sensors")

    if ts is None:  # keep a placeholder so downstream code is uniform
        ts = pd.Series(pd.NaT, index=X.index, name="timestamp")
    ts.name = "timestamp"
    return X.reset_index(drop=True), y.reset_index(drop=True), ts.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# AI4I 2020 (optional predictive-maintenance add-on, id=601)
# --------------------------------------------------------------------------- #
def load_ai4i() -> pd.DataFrame:
    """Return the AI4I 2020 predictive-maintenance dataframe (10,000 x 14)."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=601)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        print(f"[data_loader] AI4I via ucimlrepo: {df.shape}")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[data_loader] AI4I ucimlrepo failed ({exc!r}); trying mirror CSV")
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00601/ai4i2020.csv")
        dest = _download(url, config.DATA_RAW / "ai4i2020.csv")
        return pd.read_csv(dest)


if __name__ == "__main__":  # pragma: no cover — smoke test
    X, y, ts = load_secom()
    print(X.iloc[:3, :5])
    print("label counts:\n", y.value_counts())
    print("missing cells:", int(X.isna().sum().sum()))
