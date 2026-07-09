# Data Dictionary & Dataset Understanding

## 1. Dataset at a glance

| Attribute | Detail |
|---|---|
| Name | SECOM (SEmiCONductor Manufacturing) |
| Origin | A real semiconductor wafer-fabrication line; back-end (test) process data |
| Source | UCI ML Repository, dataset **179** — https://archive.ics.uci.edu/dataset/179/secom |
| Donors | Michael McCann, Adrian Johnston (2008) |
| Files | `secom.data` (sensor matrix), `secom_labels.data` (label + timestamp) |
| Rows | 1,567 wafers (one row = one production unit through the line) |
| Columns | 590 sensor/process measurements |
| Target | Pass/Fail at final electrical test |
| Domain | Industrial IoT / smart-factory process monitoring |

## 2. Variable groups

Because the 590 features are **anonymized** (released as raw signal values with no
physical names, to protect process IP), they are documented by *group* rather than
individually. In code they are named `sensor_000 … sensor_589`.

| Variable | Type | Role | Description |
|---|---|---|---|
| `sensor_000 … sensor_589` | continuous (float) | features | Individual process/metrology signals captured as each wafer passes the monitored steps. Units are undisclosed and differ across sensors (hence scaling is required). Contain missing values where a sensor did not report. |
| `label_raw` | categorical (int) | raw target | `-1` = **pass**, `+1` = **fail** at final test. |
| `fail` | binary (int) | modelling target | Re-encoded: `0` = pass, `1` = fail (fail = the positive/minority class). |
| `event_ts` | datetime | metadata | Timestamp of measurement acquisition. Used for daily-yield KPIs and ordering/drift checks — **excluded from the model matrix** (not a process signal; risk of leakage). |

> A machine-readable per-sensor profile (count, mean, std, % missing, near-constant
> flag) is generated to `data/processed/secom_feature_meta.csv` / the Hive
> `sensor_profile` table when the pipeline runs.

## 3. Key statistical characteristics (the issues cleaning must fix)

| Characteristic | Value | Consequence for analysis |
|---|---|---|
| Missing cells | ~41,951 | Imputation required; columns >50% missing dropped. |
| Near-constant sensors | ~347 | Carry no discriminative signal → variance pruning. |
| Heterogeneous scales | 590 different ranges/units | Standardisation needed for linear/distance models. |
| Class imbalance | 1,463 : 104 (~14:1) | Accuracy is misleading; use class weights / SMOTE + recall/PR-AUC. |
| Outliers / spikes | present in many sensors | Winsorize (keep rows — failures are precious). |

## 4. Context: where this data sits in the fab

A wafer travels through hundreds of deposition, etch, lithography, implant and test
steps. In-line metrology and equipment sensors log signals at many of these steps.
SECOM is a **snapshot of those back-end signals plus the final pass/fail verdict** —
precisely the data a fab would mine to ask *"what upstream signal predicts a bad
unit?"*. The intended use documented by the donors is **feature selection**: finding
the small subset of signals that actually matter, so engineers are not forced to
monitor all 590 equally.

## 5. Limitations (stated honestly)

1. **Anonymized features** — we cannot name physical steps, so deliverables are
   ranked *sensor IDs*, not "etch chamber 3 pressure". A real fab maps IDs internally.
2. **Severe imbalance & small positive class** — only 104 failures cap achievable
   precision; the project optimises for *recall* and *ranking*, and is explicit that
   absolute performance is bounded by failure scarcity.
3. **No causal claim** — feature importance is associational. A high-ranked sensor is
   a *lead to investigate*, not a proven root cause.
4. **Single line, fixed period** — patterns may not transfer to other tools/fabs or
   survive process drift; periodic retraining is assumed.
5. **Synthetic add-on** — the AI4I 2020 mini-study is *synthetic* data, included only
   to illustrate the predictive-maintenance angle; it is kept secondary to SECOM.
