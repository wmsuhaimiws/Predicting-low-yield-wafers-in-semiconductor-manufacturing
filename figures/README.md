# `figures/` — generated visualizations

All figures are produced **automatically** by running the notebook
(`notebooks/secom_yield_analysis.ipynb`) or `python -m src.eda`. They are written
here as 150-dpi PNGs. The interactive Plotly dashboard is exported to
`docs/interactive_dashboard.html`.

| File | Serves | Chart | What it answers |
|---|---|---|---|
| `01_class_balance.png` | EDA | Bar + % annotation | How severe is the pass/fail imbalance? |
| `02_missingness_map.png` | EDA | Missingness heatmap / bar | Where are the ~42k missing cells concentrated? |
| `03_missingness_hist.png` | EDA | Histogram of per-column missing % | How many columns exceed the drop threshold? |
| `04_variance_distribution.png` | EDA | Log-variance histogram | How many sensors are near-constant? |
| `05_feature_distributions.png` | EDA | Small-multiple KDEs, pass vs fail | Which signals shift between pass and fail? |
| `06_correlation_heatmap.png` | EDA | Clustered correlation heatmap | How redundant is the sensor set? |
| `07_top_feature_importance.png` | EDA/Insights | Horizontal bar | Which sensor IDs predict failure most? |
| `08_passfail_boxplots.png` | EDA/Insights | Box/violin per top sensor | How do top signals separate the two classes? |
| `09_yield_pareto.png` | Creativity | Pareto | Do a few signals drive most of the predictive signal? |
| `10_pr_roc_curves.png` | Insights | PR + ROC curves | How well does the recall-focused model perform? |
| `11_ai4i_failure_modes.png` | Creativity | Grouped bar | Add-on: distribution of the 5 failure modes. |

## Design principles applied (Few / "Evergreen" visualization)

- **Right chart for the data:** distributions → KDE/box; relationships → heatmap;
  rankings → horizontal bars/Pareto; proportions → annotated bars.
- **High data-ink ratio:** no 3-D, no gradients, gridlines muted, direct labels.
- **Deliberate, restrained colour:** a single accent (`#C0392B`) reserved for the
  *fail* class so the eye always reads "red = risk"; pass is neutral grey/blue.
- **Every figure is captioned** in the notebook and tied back to a decision.
