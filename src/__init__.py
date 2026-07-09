"""
SECOM Yield Optimization — source package.

Modular pipeline for the MSc Data Management final project.

Modules
-------
config         Central paths, constants and tunable thresholds.
data_loader    Robust acquisition of SECOM (id=179) and AI4I (id=601).
preprocessing  Missing-value, low-variance, outlier, scaling and imbalance logic.
eda            Reproducible exploratory figures.
modeling       Imbalance-aware training, evaluation and feature importance.
viz_extras     Creativity extras: interactive Plotly dashboard + yield Pareto.
ai4i           Optional predictive-maintenance mini-study (creativity add-on).
run_pipeline   One-command end-to-end orchestrator.
"""

__version__ = "1.0.0"
