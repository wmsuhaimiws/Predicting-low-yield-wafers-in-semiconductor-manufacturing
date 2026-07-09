# SECOM Yield Optimization — Predicting Low-Yield Wafers in Semiconductor Manufacturing

> **Industrial IoT (IIoT) · Smart-Factory Yield Optimization · Semiconductor Packaging & Test**
> An end-to-end data-management project: from raw fab-sensor telemetry to
> actionable, yield-improving recommendations for a process-engineering team.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/data-UCI%20research%20use-lightgrey)

---

## 1. Why this industry? (Dataset Selection & Relevance)

Semiconductor manufacturing is one of the highest-value, highest-demand domains a
data scientist can work in — and it is **strategically central to Malaysia**, where
this project is based:

- Malaysia is the world's **6th-largest semiconductor exporter** and handles roughly
  **13% of global semiconductor packaging, assembly and test (OSAT)** — the exact
  back-end process stage that SECOM data comes from.
- The sector exported about **US$130 billion in 2024**, and Malaysia holds ~**7% of
  the global semiconductor market**, with a national target to nearly double that
  share by 2029.
- **Penang — "the Silicon Valley of the East"** — alone accounts for ~5% of global
  semiconductor sales and hosts 350+ multinationals (Intel's >US$7B advanced
  packaging investment among them) plus thousands of supplier SMEs.

In a high-volume fab, **yield** (the fraction of wafers that pass final test) is the
dominant lever on per-unit cost. A single percentage point of yield can be worth
millions annually. That makes *"which process signals predict a failing wafer?"* a
genuinely high-income, high-impact data-science question — exactly the kind of
problem this project tackles.

## 2. The business problem

> **Which process signals predict low-yield (fail) wafers, and how can the fab use
> them to raise yield and cut per-unit cost?**

Every wafer passes through hundreds of process steps, each monitored by sensors.
Some wafers fail final electrical test — lost yield and wasted cost. If we can
identify *which sensor signals* are most associated with failure, process engineers
know where to investigate first. Because SECOM's features are anonymized, we treat
each as a **sensor ID** and deliver a ranked list of the **top predictive signals**
as engineering leads — which matches the dataset's documented intended use.

### Objectives

1. Build a clean, reproducible, leakage-free pipeline over real fab-sensor data.
2. Handle the dataset's real-world defects: missing values, near-constant sensors,
   outliers, heterogeneous scales and a severe ~14:1 class imbalance.
3. Train an **imbalance-aware** classifier optimised for **catching failures**
   (recall / PR-AUC), not headline accuracy.
4. Surface the **top predictive sensor IDs** as prioritised engineering leads.
5. Translate the findings into **actionable recommendations** for a fab/process team.

## 3. Dataset (Data Collection & Understanding)

**UCI SECOM** — real semiconductor wafer-fab process measurements.

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset **179** — https://archive.ics.uci.edu/dataset/179/secom |
| Records | 1,567 production wafers |
| Features | 590 anonymized continuous sensors |
| Label | `-1` = pass, `+1` = fail (final test) |
| Timestamp | per-record acquisition time |
| Missing values | ~41,951 cells |
| Near-constant columns | ~347 low/zero-variance sensors |
| Class balance | 1,463 pass : 104 fail (**~14:1**) |

Full variable documentation is in [`docs/data_dictionary.md`](docs/data_dictionary.md);
known limitations (anonymized features, imbalance, the synthetic add-on) are stated
honestly there and in [`docs/methodology.md`](docs/methodology.md).

**Optional creativity add-on:** a secondary IIoT predictive-maintenance mini-study on
the **UCI AI4I 2020** dataset (id 601, 10,000 × 14, five labelled failure modes) — see
[`src/ai4i.py`](src/ai4i.py).

## 4. Tech stack (Application of Tools)

| Layer | Tools |
|---|---|
| Analysis & ML | Python, pandas, NumPy, scikit-learn, imbalanced-learn |
| Visualization | matplotlib, seaborn, Plotly (interactive dashboard) |
| Big-data ingest/aggregation | **Apache Pig**, **Apache Hive** (real Hadoop + via Spark SQL), **DuckDB** — four engines, one job |
| Environment | Google Colab / Jupyter; fully pinned `requirements.txt` |
| Data acquisition | `ucimlrepo` with direct-CSV / Kaggle-mirror fallbacks |
| Presentation | 11-slide deck (`.pptx`) + one-page executive-summary PDF |

The big-data component (`src/bigdata/`, narrative in
[`docs/bigdata_hive_pig.md`](docs/bigdata_hive_pig.md)) shows how the same ingest +
aggregation scales when the fab streams telemetry that no longer fits in memory.
**Four runnable notebooks** demonstrate the *same* per-sensor / pass-fail aggregation
in **Pig**, **Hive** (via Spark SQL *and* real Hadoop+Hive) and **DuckDB**, and the
doc explains *when* to reach for each engine versus in-memory pandas.

## 5. Repository map (Repository Professionalism)

```
secom-yield-optimization/
├── README.md                      ← you are here
├── LICENSE                        ← MIT (code) + data-use notice
├── requirements.txt               ← pinned, reproducible environment
├── .gitignore
├── SECOM_Yield_Optimization.pddf  ← PDF presentation 11 slides
├── notebooks/
│   ├── secom_yield_analysis.ipynb       ← analysis & modelling (pandas/scikit-learn)
│   ├── bigdata_demo_pig_hive.ipynb      ← Pig (local mode) + HiveQL via Spark SQL
│   ├── bigdata_demo_pig_duckdb.ipynb    ← Pig (local mode) + DuckDB (zero-Java)
│   └── bigdata_demo_hadoop_hive.ipynb   ← real Hadoop + Hive (most authentic)
├── src/
│   ├── config.py                  ← all paths & tunable thresholds
│   ├── data_loader.py             ← robust SECOM/AI4I loading (3 fallbacks)
│   ├── preprocessing.py           ← cleaning with before/after audit
│   ├── eda.py                     ← 8 EDA figures
│   ├── modeling.py                ← imbalance-aware models + feature importance
│   ├── viz_extras.py              ← Plotly dashboard + yield Pareto
│   ├── ai4i.py                    ← predictive-maintenance add-on
│   ├── run_pipeline.py            ← one-command end-to-end orchestrator
│   └── bigdata/
│       ├── melt_to_long.py        ← ETL: wide → long sensor-event stream
│       ├── secom_ingest.hql       ← Hive ingest + aggregation
│       └── secom_aggregate.pig    ← Pig Latin equivalent
├── figures/                       ← generated PNGs (+ design notes in README)
└── docs/
    ├── data_dictionary.md         ← variables, types, limitations
    ├── methodology.md             ← every cleaning/modelling choice justified
    ├── bigdata_hive_pig.md        ← when Hive/Pig vs pandas
    ├── insights.md                ← data-driven insights + semiconductor context
    ├── recommendations.md         ← actionable fab recommendations
    ├── executive_summary.md       ← one-page summary for leadership
    ├── executive_summary.pdf      ← one-page summary, print-ready (A4)
    ├── pipeline_diagram.svg       ← methodology diagram
    └── github_setup.md            ← exact publish/push steps
```

## 6. Reproduce in 5 minutes (Code Quality & Reproducibility)

### Option A — Google Colab (recommended, zero setup)

1. Open `notebooks/secom_yield_analysis.ipynb` in Colab.
2. `Runtime → Run all`. The first cell `pip install`s the dependencies and fetches
   SECOM via `ucimlrepo` (with automatic fallbacks). No manual download, no local
   file paths.

### Option B — Local

```bash
git clone <your-fork-url> secom-yield-optimization
cd secom-yield-optimization
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the entire pipeline (load → clean → EDA → model → extras):
python -m src.run_pipeline
```

Outputs: cleaned data in `data/processed/`, all figures in `figures/`, the
interactive dashboard at `docs/interactive_dashboard.html`.

### Option C — Big-data component (Hive + Pig)

Easiest: open **`notebooks/bigdata_demo_pig_hive.ipynb`** (Pig + HiveQL via Spark) or
**`notebooks/bigdata_demo_pig_duckdb.ipynb`** (Pig + DuckDB, zero-Java) in Colab and
**Run all** — no cluster needed. For the most authentic path, `bigdata_demo_hadoop_hive.ipynb`
installs real Hadoop + Hive and runs the HiveQL for real (heavier, ~5-10 min). On a real cluster/laptop:

```bash
python -m src.bigdata.melt_to_long          # land long-format TSVs
hive -f src/bigdata/secom_ingest.hql        # Hive aggregation
pig  -x local -param LONG_INPUT=... -param LABELS_INPUT=... -param OUT_DIR=... \
     -f src/bigdata/secom_aggregate.pig     # Pig equivalent (parametrized paths)
```

## 7. Headline results & where to read more

- **Cleaning** removes high-missing and near-constant sensors and imputes the rest;
  the before/after audit prints when you run the pipeline (and is summarised in
  `docs/methodology.md`).
- **Modelling** compares cost-sensitive learning vs SMOTE across Logistic Regression
  and Random Forest, reported on **recall / PR-AUC for the fail class** — the honest
  metrics under 14:1 imbalance.
- **Top predictive sensor IDs** are delivered as prioritised engineering leads, with
  a **yield Pareto** showing the "vital few" signals.
- Read the story end-to-end: **[insights](docs/insights.md) → [recommendations](docs/recommendations.md) → [executive summary](docs/executive_summary.md)**.
- **Present it:** an 11-slide viva deck (`SECOM_Yield_Optimization.pptx`) and a
  print-ready **[one-page executive-summary PDF](docs/executive_summary.pdf)**.

## 8. Honest limitations

- Features are **anonymized**, so we report sensor *IDs*, not physical process names —
  the fab would map IDs back to steps internally.
- Only **104 failures** exist; absolute performance is bounded by this scarcity, so we
  emphasise ranking and recall over precision.
- The **AI4I** add-on is **synthetic** and kept strictly secondary to SECOM.

## 9. License & attribution

Data © UCI Machine Learning Repository (research use). Please cite:

> Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository.* Irvine, CA:
> University of California, School of Information and Computer Science.
> SECOM: McCann, M. & Johnston, A. (2008).

---

### Sources for the industry figures (Section 1)

- [Why Malaysia Leads in Chip Packaging and Testing Exports — One Union Solutions](https://oneunionsolutions.com/blog/malaysia-becomes-a-global-hub-for-semiconductor-packaging-and-testing-exports/)
- [Malaysia's semiconductor supply chain: A regional powerhouse — Switzerland Global Enterprise](https://www.s-ge.com/en/publication/fact-sheet/2025-e-mem-malaysia-ct6-semiconductor-chain-value-supply)
- [Securing Malaysia's position in the global semiconductor supply chain — MIDA](https://www.mida.gov.my/mida-news/securing-malaysias-position-in-the-global-semiconductor-supply-chain/)
- [Malaysia's Semiconductor Growth: Can It Move Up the Value Chain? — ASEAN Briefing](https://www.aseanbriefing.com/news/malaysias-semiconductor-growth-can-it-move-up-the-value-chain/)
