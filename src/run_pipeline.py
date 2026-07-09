"""
run_pipeline.py — one-command, end-to-end orchestrator.

    python -m src.run_pipeline

Runs the full data-management workflow: load -> clean -> EDA -> model ->
feature importance -> creativity extras, printing a before/after audit and the
model comparison, and writing every figure + the cleaned dataset to disk.
"""
from __future__ import annotations
import warnings

import pandas as pd

from . import config, eda, modeling, preprocessing, viz_extras
from .data_loader import load_secom

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 120)


def main(run_ai4i: bool = True):
    print("=" * 70)
    print("SECOM YIELD OPTIMIZATION — full pipeline")
    print("=" * 70)

    # 1. LOAD
    X_raw, y_raw, ts = load_secom()

    # 2. CLEAN
    X_clean, y, report = preprocessing.clean_secom(X_raw, y_raw)
    print("\n--- Cleaning before/after summary ---")
    print(report.to_frame().to_string(index=False))
    print(f"\nClass imbalance ratio: {preprocessing.imbalance_ratio(y):.1f} : 1")

    out_csv = config.DATA_PROCESSED / "secom_clean.csv"
    pd.concat([X_clean, y], axis=1).to_csv(out_csv, index=False)
    print(f"Saved cleaned dataset -> {out_csv}")

    # 3. MODEL
    X_tr, X_te, y_tr, y_te = modeling.make_split(X_clean, y)
    results = modeling.train_models(X_tr, X_te, y_tr, y_te)
    table = modeling.results_table(results)
    print("\n--- Model comparison (recall/PR-AUC focused) ---")
    print(table.to_string(index=False))

    best = max(results, key=lambda r: (r.pr_auc, r.recall_fail))
    print(f"\nBest model: {best.name}")
    print(best.report_text)
    tuned = modeling.tune_threshold(best, X_te, y_te)
    print(f"Recall-optimised threshold: {tuned}")

    # 4. FEATURE IMPORTANCE
    imp_top, imp_full = modeling.top_feature_importance(
        X_tr, y_tr, feature_names=X_clean.columns)
    print("\n--- Top predictive sensors (engineering leads) ---")
    print(imp_top[["sensor", "importance"]].to_string(index=False))

    # 5. FIGURES + CREATIVITY EXTRAS
    eda.run_all(X_raw, X_clean, y, top_sensors=imp_top["sensor"].tolist(),
                imp_top=imp_top)
    viz_extras.yield_pareto(imp_full)
    viz_extras.build_dashboard(X_clean, y, imp_full, table)

    # 6. OPTIONAL AI4I MINI-STUDY
    if run_ai4i:
        try:
            from . import ai4i
            metrics, _ = ai4i.run_ai4i_study()
            print(f"\n[AI4I add-on] ROC-AUC = {metrics['roc_auc']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[AI4I add-on] skipped: {exc}")

    print("\nDone. Figures in figures/, dashboard in docs/, clean data in data/processed/.")
    return {"report": report, "results": table, "top_features": imp_top}


if __name__ == "__main__":
    main()
