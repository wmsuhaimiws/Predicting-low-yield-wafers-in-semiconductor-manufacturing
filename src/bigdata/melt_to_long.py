"""
melt_to_long.py — ETL landing step for the big-data component.

At true fab scale, sensor telemetry arrives as an append-only event stream:

        wafer_id , sensor_id , reading , event_ts

This script reshapes the SECOM wide matrix into that long/tall event format and
lands it as partition-friendly TSV in `data/processed/long/`, ready to be
exposed to Hive (CREATE EXTERNAL TABLE) or Pig (LOAD).

Run:  python -m src.bigdata.melt_to_long
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from .. import config
from ..data_loader import load_secom


def melt_secom(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (config.DATA_PROCESSED / "long")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_raw, ts = load_secom()
    X = X.copy()
    X.insert(0, "wafer_id", range(len(X)))

    # wafer-level label/timestamp table (small, one row per wafer)
    labels = pd.DataFrame({
        "wafer_id": range(len(X)),
        "label_raw": y_raw.values,                       # -1 pass / +1 fail
        "fail": (y_raw.values == config.LABEL_FAIL_RAW).astype(int),
        "event_ts": ts.values,
    })
    labels_path = out_dir / "secom_labels.tsv"
    labels.to_csv(labels_path, sep="\t", index=False, header=False)

    # long sensor-event table (one row per wafer x sensor)
    long = X.melt(id_vars="wafer_id", var_name="sensor_id", value_name="reading")
    long["sensor_id"] = long["sensor_id"].str.replace("sensor_", "", regex=False)
    long = long.dropna(subset=["reading"])
    long_path = out_dir / "secom_sensor_long.tsv"
    long.to_csv(long_path, sep="\t", index=False, header=False)

    summary = {
        "wafers": len(labels),
        "sensor_events": len(long),
        "labels_file": str(labels_path),
        "long_file": str(long_path),
    }
    print(f"[melt] wrote {summary['sensor_events']:,} sensor events -> {long_path}")
    print(f"[melt] wrote {summary['wafers']:,} wafer labels   -> {labels_path}")
    return summary


if __name__ == "__main__":
    melt_secom()
