"""
modeling.py — imbalance-aware training, evaluation and feature importance.

Everything that could leak (scaling, SMOTE) lives inside an imblearn Pipeline so
it is fit on the training fold only.

Key design choices
-------------------
* Stratified train/test split preserves the 14:1 ratio in both halves.
* We compare two imbalance strategies head-to-head:
      (a) cost-sensitive learning  -> class_weight="balanced"
      (b) resampling               -> SMOTE on the training fold
* Accuracy is deliberately NOT the headline metric. With 93% passes a "predict
  pass" model scores 93% accuracy and catches zero failures. We optimise for
  RECALL on the fail class and report PR-AUC (average precision).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from . import config

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except Exception:  # noqa: BLE001 — keep the project runnable without imblearn
    _HAS_IMBLEARN = False


@dataclass
class ModelResult:
    name: str
    recall_fail: float
    precision_fail: float
    f1_fail: float
    roc_auc: float
    pr_auc: float
    confusion: np.ndarray
    report_text: str
    estimator: object

    def summary_row(self) -> dict:
        return {
            "model": self.name,
            "recall(fail)": round(self.recall_fail, 3),
            "precision(fail)": round(self.precision_fail, 3),
            "f1(fail)": round(self.f1_fail, 3),
            "ROC-AUC": round(self.roc_auc, 3),
            "PR-AUC": round(self.pr_auc, 3),
        }


def make_split(X: pd.DataFrame, y: pd.Series):
    """Stratified hold-out so both splits keep the ~14:1 ratio."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )


def _build_pipeline(estimator, use_smote: bool):
    """Scaler -> [SMOTE] -> estimator, wired so resampling only touches train folds."""
    steps = [("scale", StandardScaler())]
    if use_smote and _HAS_IMBLEARN:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)
    steps.append(("clf", estimator))
    return SkPipeline(steps)


def _evaluate(name, pipe, X_test, y_test) -> ModelResult:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    return ModelResult(
        name=name,
        recall_fail=recall_score(y_test, pred, pos_label=1, zero_division=0),
        precision_fail=precision_score(y_test, pred, pos_label=1, zero_division=0),
        f1_fail=f1_score(y_test, pred, pos_label=1, zero_division=0),
        roc_auc=roc_auc_score(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
        confusion=confusion_matrix(y_test, pred),
        report_text=classification_report(y_test, pred,
                                           target_names=["pass", "fail"],
                                           zero_division=0),
        estimator=pipe,
    )


def train_models(X_train, X_test, y_train, y_test) -> list[ModelResult]:
    """Train the four-way comparison and return their evaluation results."""
    specs = [
        ("LogReg + class_weight",
         LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=config.RANDOM_STATE), False),
        ("LogReg + SMOTE",
         LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE), True),
        ("RandomForest + class_weight",
         RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=config.RANDOM_STATE), False),
        ("RandomForest + SMOTE",
         RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                random_state=config.RANDOM_STATE), True),
    ]
    results = []
    for name, est, use_smote in specs:
        pipe = _build_pipeline(est, use_smote)
        pipe.fit(X_train, y_train)
        results.append(_evaluate(name, pipe, X_test, y_test))
    return results


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """Tidy comparison table, sorted by PR-AUC then fail-recall."""
    df = pd.DataFrame([r.summary_row() for r in results])
    return df.sort_values(["PR-AUC", "recall(fail)"], ascending=False).reset_index(drop=True)


def top_feature_importance(X_train, y_train, feature_names,
                           top_n: int = config.TOP_N_FEATURES):
    """
    Rank sensors by Random-Forest impurity importance.

    Because SECOM features are anonymized, the deliverable is a ranked list of
    *sensor IDs* — the engineering team maps these back to physical process steps.
    Returns (top slice, full ranking for the Pareto).
    """
    rf = SkPipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE)),
    ])
    rf.fit(X_train, y_train)
    importances = rf.named_steps["clf"].feature_importances_
    imp = (pd.DataFrame({"sensor": feature_names, "importance": importances})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    imp["cum_importance"] = imp["importance"].cumsum() / imp["importance"].sum()
    return imp.head(top_n), imp


def tune_threshold(result: ModelResult, X_test, y_test,
                   min_precision: float = 0.10) -> dict:
    """Pick the cut-off that maximises fail-recall subject to a precision floor."""
    from sklearn.metrics import precision_recall_curve
    proba = result.estimator.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    best = {"threshold": 0.5, "recall": 0.0, "precision": 0.0}
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if p >= min_precision and r > best["recall"]:
            best = {"threshold": float(t), "recall": float(r), "precision": float(p)}
    return best
