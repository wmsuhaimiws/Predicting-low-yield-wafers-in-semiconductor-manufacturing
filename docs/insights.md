# Insights, Interpretation & Critical Thinking

The numbers below in **bold-fixed form** (imbalance, missingness total, near-constant
count) are intrinsic properties of SECOM and hold every run. Model-performance and
specific top-sensor IDs are produced by `python -m src.run_pipeline` and printed in
the notebook; this document interprets the *patterns* and explains the **why** in
semiconductor terms — which is what turns numbers into engineering value.

## Insight 1 — The problem is rare-event detection, not classification

Failures are **104 of 1,567 wafers (~6.6%, a 14:1 imbalance)**. The central
analytical consequence: **accuracy is a trap.** A "predict pass" model scores ~93%
accuracy while catching *zero* failures. Reframing the task as **rare-event
detection** — optimise recall/PR-AUC, tune the decision threshold — is the single
most important decision in the project.

*Why it matters in a fab:* the asymmetry of cost is real. A missed failure can mean a
defective unit escaping to assembly or a customer return (expensive, reputational);
a false alarm costs an engineer an hour. The model should be tuned to that asymmetry,
not to a symmetric accuracy score.

## Insight 2 — Most sensors are noise; the signal is concentrated

Of 590 sensors, **~347 are near-constant** and a further tail is mostly missing.
After cleaning, the predictive signal concentrates in a **small minority of
sensors**, which the **yield Pareto** (`figures/09_yield_pareto.png`) makes visible:
a handful of sensor IDs account for the bulk of cumulative importance.

*Why it matters:* this is exactly the dataset's documented intended use —
**feature selection**. A fab cannot give 590 signals equal engineering attention.
Telling the team *"these ~10–15 signals carry most of the failure information"*
converts an unmanageable monitoring problem into a focused SPC (statistical process
control) watch-list. It also means cheaper data collection: low-value sensors can be
sampled less aggressively.

## Insight 3 — Missingness is structural, not random

The **~41,951 missing cells** are not spread uniformly; they cluster in specific
sensors (the missingness map, `figures/02_missingness_map.png`, shows columns that
are blank for long contiguous runs of wafers).

*Why it matters:* contiguous missing blocks usually mean a **sensor or logging
channel was offline for a period**, not random dropout. That is itself an
operational finding — a data-management/instrumentation gap to fix at source. It also
justifies the cleaning choice: drop the worst offenders (can't honestly impute >50%)
and median-impute the rest, rather than pretending the gaps are noise.

## Insight 4 — Top signals *separate* the classes, but imperfectly

The pass/fail box-plots and KDEs (`figures/05`, `figures/08`) show the top sensors
**shift** between pass and fail wafers — but with **heavy overlap**. No single sensor
is a clean discriminator.

*Why it matters:* failure in a fab is typically **multivariate and interaction-
driven** — a marginal temperature *combined with* a borderline pressure, not either
alone. This is why the Random Forest (which models interactions) is informative and
why the recommendation is a **multi-signal monitoring rule**, not a single-sensor
red line. It also tempers expectations: PR-AUC will be modest because the classes
genuinely overlap, which is honest and expected for real process data.

## Insight 5 — Class-weighting vs SMOTE is a real trade-off, not a formality

The four-way comparison (`results_table`) exists because the two imbalance remedies
behave differently: **class weights** keep the data real but lean on the model;
**SMOTE** balances the training fold by synthesising minority points, which can lift
recall but risks manufacturing implausible "wafers" in sparse sensor space.

*Why it matters:* reporting both — and selecting on **PR-AUC then recall** — shows the
choice was made on evidence, not default. For a fab, the recall-optimised threshold
(printed by `tune_threshold`) is the knob management actually turns: *"how many real
failures are we willing to chase false alarms to catch?"*

## Insight 6 — From signals to yield economics

Because yield is the dominant cost lever in high-volume packaging/test, surfacing the
**vital-few signals** has direct economic translation: tightening control on those
specific steps is where a yield point is most cheaply won. The project therefore ends
not at a confusion matrix but at a **prioritised, finding-supported action list**
(see [`recommendations.md`](recommendations.md)) — connecting the data story back to
the per-unit-cost question that motivated it.

---

### Critical caveats (intellectual honesty)

- **Associational, not causal.** A high-importance sensor is a *lead*, confirmed only
  by a designed experiment (DOE) or engineering review on the line.
- **Bounded by 104 failures.** Absolute precision is capped by failure scarcity;
  more labelled failures (or active learning on uncertain wafers) would help most.
- **Drift.** A single line over a fixed window — the watch-list must be **re-validated
  on a rolling basis** as recipes and tools change.
