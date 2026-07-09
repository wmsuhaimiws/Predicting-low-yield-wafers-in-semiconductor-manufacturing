# Methodology — Cleaning, Preprocessing & Modelling

This document gives the **written justification** for every transformation. Each
step is implemented in `src/preprocessing.py` / `src/modeling.py` and prints a
**before/after summary** when the pipeline runs (`python -m src.run_pipeline`).

## Guiding principle: no leakage

Operations whose statistics depend on the data are split into two classes:

- **Structural / row-independent** (drop high-missing columns, drop near-constant
  columns) — applied to the whole frame; they do not "learn" from the label.
- **Statistic-fitting** (scaling, SMOTE) — fit **inside an imbalanced-learn
  Pipeline on the training fold only**, so the test set never influences them.

This is the single most important reproducibility/validity decision in the project.

---

## Step-by-step cleaning

| # | Step | What & why | Before → After (expected) |
|---|---|---|---|
| 1 | **Encode label** | Map `-1/​+1` → `0/1` with **fail = 1**. Making the rare failure the positive class makes recall/precision/PR-AUC read as "how well do we catch failures". | labels `{-1,1}` → `{0,1}` |
| 2 | **Drop high-missing sensors** (>50%) | A sensor silent for most wafers can't be imputed honestly — filling >half its values fabricates signal. Dropping is safer. | 590 → ~560 cols* |
| 3 | **Median imputation** | Fill remaining NaNs with the column **median**. Median (not mean) is robust to the skew and spikes typical of sensor data, so imputed values stay physically plausible. Fitted medians are stored for reuse on new wafers. | ~41,951 missing → **0** |
| 4 | **Drop near-constant sensors** | Remove columns that are constant or whose single dominant value covers >99% of rows. They add dimensionality and multicollinearity but zero discriminative power. SECOM documents ~347 such columns. | ~560 → ~410 cols* |
| 5 | **Winsorize outliers** (1st/99th pct) | **Clip, don't delete.** Extreme readings may be genuine excursions preceding failure — exactly the signal we want. With only 104 failures we never drop rows; clipping just limits a glitch's leverage on scaling/linear models. | extreme tails capped |
| 6 | **Standardise** (inside model pipeline) | 590 sensors on different units → `StandardScaler` so no feature dominates by scale. Fit on train fold only. | mean≈0, std≈1 per feature |
| 7 | **Address 14:1 imbalance** | Two strategies compared head-to-head: **class weights** (cost-sensitive) and **SMOTE** (synthesise minority examples, train fold only). | effective 1:1 during training |

\* exact counts are printed by the `CleaningReport` audit at runtime; they depend on
the chosen thresholds in `src/config.py` and are reproducible.

### Why these thresholds (all centralised in `config.py`)

- `MISSING_COL_THRESHOLD = 0.50` — a balance between keeping sensors and not
  imputing more than half a column. Easily changed and re-run.
- `DOMINANT_VALUE_THRESHOLD = 0.99` — captures "effectively constant" without
  discarding sensors that have genuine but low variation.
- `WINSOR_LIMITS = (0.01, 0.99)` — symmetric, conservative; tames glitches without
  reshaping the bulk of the distribution.

---

## Modelling design

### Validation
Stratified 75/25 hold-out preserves the ~14:1 ratio in both splits. All
preprocessing that fits parameters happens inside the pipeline, so the hold-out is
a clean estimate.

### Models compared
| Model | Imbalance strategy |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Logistic Regression | SMOTE |
| Random Forest | `class_weight="balanced_subsample"` |
| Random Forest | SMOTE |

Logistic Regression gives an interpretable linear baseline; Random Forest captures
non-linear sensor interactions and yields the feature-importance ranking.

### Metrics — and why **not accuracy**
With 93% passes, a model that predicts "pass" for everything scores **93% accuracy
and catches zero failures**. We therefore report and select on:

- **Recall (fail)** — share of true failures caught (the cost of a miss is a bad
  unit shipped / scrapped late).
- **Precision (fail)** — share of failure alerts that are real (the cost of a false
  alarm is wasted engineering time).
- **F1 (fail)** and **PR-AUC / average precision** — threshold-independent summaries
  appropriate under heavy imbalance (PR-AUC is far more informative than ROC-AUC
  here because the negative class dominates).

### Threshold tuning
The default 0.5 cut-off is rarely optimal under imbalance. `tune_threshold` picks
the probability cut-off that **maximises fail-recall subject to a precision floor**,
making the fab's "miss vs false-alarm" trade-off explicit and adjustable.

### Feature importance → engineering leads
Random-Forest impurity importance ranks the sensors; the top *N* (default 15) are
delivered as **prioritised sensor IDs** with a cumulative-importance **Pareto** to
show how few signals carry most of the predictive power. This is associational, not
causal — each is a *lead to investigate*, in line with SECOM's documented
feature-selection intent.
