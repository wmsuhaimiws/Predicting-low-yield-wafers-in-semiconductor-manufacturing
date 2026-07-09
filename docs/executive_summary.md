# Executive Summary — SECOM Yield Optimization (one page)

**Audience:** fab / operations leadership · **Prepared by:** Data Management project ·
**Domain:** Semiconductor packaging & test (Industrial IoT)

---

### The question
In a high-volume semiconductor line, **yield** is the dominant driver of per-unit
cost. We asked: *which process signals predict a failing wafer, and how can the fab
use them to raise yield?* Data: **UCI SECOM** — 1,567 real wafers × 590 anonymized
process sensors, with a pass/fail final-test verdict.

### What makes it hard
- **Severe imbalance:** only **104 failures vs 1,463 passes (~14:1)** — so accuracy
  is meaningless; we must optimise for *catching failures*.
- **Messy telemetry:** **~42,000 missing values** and **~347 near-constant sensors**
  that carry no signal.
- **Anonymized features:** we deliver ranked **sensor IDs** as engineering leads.

### What we did
A clean, reproducible, **leakage-safe** pipeline: drop unreliable/constant sensors →
median-impute → winsorize outliers → standardise → handle imbalance with **class
weights and SMOTE** → compare Logistic Regression and Random Forest, judged on
**recall and PR-AUC for the fail class** (not accuracy). Feature importance, shown as
a **yield Pareto**, surfaces the vital-few signals. The same ingest + aggregation is
expressed in **Hive and Pig** for fab-scale data.

### What we found
1. The signal is **concentrated** — a small minority of sensors carry most of the
   predictive power; the other ~570 are largely noise.
2. Failure is **multivariate** — top sensors shift between pass and fail but overlap
   heavily; no single red line works, so a multi-signal rule is needed.
3. Missingness is **structural** (sensors offline for periods) — a fixable
   data-quality issue, not random noise.

### What to do (highest-leverage first)
1. **Tighten SPC on the vital-few sensor IDs** — the cheapest yield win.
2. **Run the model as a recall-tuned early-warning screen** for at-risk wafers, with
   an engineer in the loop (prioritise attention, don't auto-scrap).
3. **Fix the offline-sensor instrumentation gaps** at source.
4. **Confirm leads with designed experiments** before changing recipes.
5. **Retrain on a rolling cadence**; use active learning to grow the scarce failure
   set and lift the performance ceiling.

### The takeaway
> Stop optimising for accuracy and stop monitoring all 590 signals equally. Catch the
> rare failure, watch the vital few, and the data pipeline becomes a durable lever on
> yield and per-unit cost.

*Full detail: [README](../README.md) · [methodology](methodology.md) ·
[insights](insights.md) · [recommendations](recommendations.md).*
