# Recommendations & Conclusion

Each recommendation is **tied to a finding** from [`insights.md`](insights.md) and
framed for a real fab / process-engineering audience.

## Recommendations for the process-engineering team

1. **Stand up a "vital-few" SPC watch-list.**
   *Finding:* signal concentrates in ~10–15 sensor IDs (Insight 2, yield Pareto).
   *Action:* place the top-ranked sensor IDs under tightened statistical process
   control with automated alarms, instead of monitoring all 590 equally. This is the
   cheapest yield win available and directly attacks per-unit cost.

2. **Deploy the model as a recall-tuned early-warning screen, not an auto-reject.**
   *Finding:* failure is multivariate with heavy class overlap (Insight 4); cost of a
   miss ≫ cost of a false alarm (Insight 1).
   *Action:* run the classifier at the **recall-optimised threshold** (from
   `tune_threshold`) to flag at-risk wafers for engineering review / extra inspection.
   Keep a human in the loop — the tool prioritises attention, it does not scrap units.

3. **Fix the instrumentation gaps at source.**
   *Finding:* missingness is structural — contiguous blank runs imply offline sensors
   (Insight 3).
   *Action:* treat the high-missing sensors as a **data-quality maintenance ticket**.
   Better telemetry coverage will improve every downstream model and is itself an
   uptime/operational improvement.

4. **Confirm leads with designed experiments before changing recipes.**
   *Finding:* importance is associational, not causal (Insight 6 caveats).
   *Action:* take the top sensor IDs into a **DOE / engineering root-cause review** on
   the line. Only act on the steps that survive that scrutiny — avoids chasing
   spurious correlations.

5. **Retrain and re-rank on a rolling cadence; grow the failure set.**
   *Finding:* single line/period; performance bounded by 104 failures (Insight 5/6).
   *Action:* schedule **periodic retraining** to handle process drift, and use
   **active learning** (label the wafers the model is most unsure about) to enrich the
   scarce failure class — the highest-leverage way to raise the ceiling on accuracy.

6. **Push recurring aggregation into the warehouse (Hive/Pig).**
   *Finding:* at fab scale telemetry won't fit in memory (big-data section).
   *Action:* run the per-sensor profiling and pass/fail aggregation as scheduled
   **Hive/Pig** jobs on the streamed long-format data; analysts pull only the compact
   extract for modelling. Cleaner, cheaper, and production-ready.

## Conclusion

SECOM reframes a yield problem as **rare-event detection over noisy, imbalanced,
partially-instrumented sensor data** — a faithful miniature of the real challenge in
Malaysia's packaging-and-test fabs. The disciplined pipeline (honest cleaning →
leakage-safe, imbalance-aware modelling → recall-focused evaluation → Pareto-ranked
signals) turns 590 anonymous sensors into a **short, prioritised watch-list** an
engineering team can act on tomorrow.

The decisive shift is one of mindset: **optimise for catching the rare failure, not
for headline accuracy, and prioritise the vital-few signals rather than monitoring
everything.** Do that, fix the instrumentation gaps, confirm leads with DOE, and keep
the model fresh — and the data pipeline becomes a durable, compounding lever on yield
and per-unit cost.
