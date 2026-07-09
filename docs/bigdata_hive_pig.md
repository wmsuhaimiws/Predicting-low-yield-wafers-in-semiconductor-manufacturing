# Big-data component — Apache Hive & Apache Pig

> **TL;DR** — SECOM fits in a pandas DataFrame, so for *this* 1,567-row sample
> pandas is the right tool. The Hive and Pig scripts in `src/bigdata/` show how
> the **same ingest + aggregation** scales when the fab streams the data for real:
> millions of wafers × 590 sensors × many timestamps per day, which no longer
> fits in memory. This is the core of a sound data-management workflow:
> picking the engine that matches the data volume.

## The data-management workflow

```
              in-memory (this sample)                  fab-scale (production)
              -----------------------                  ----------------------
secom.data ─► pandas DataFrame ─► clean ─► model       sensor stream ─► HDFS
 (1.5k×590)        (src/*.py)                              │ (TB/day, long format)
                                                           ▼
                                        melt_to_long.py lands (wafer, sensor, reading)
                                                           ▼
                                        Hive / Pig: GROUP BY sensor_id, JOIN labels,
                                        per-sensor profile + pass/fail aggregates
                                                           ▼
                                        small aggregated extract ─► pandas ─► model
```

Hive/Pig do the **heavy reduction** (billions of rows → a few thousand per-sensor
summary rows); pandas/scikit-learn then model the small extract. The two engines
are complementary, not competing.

## Why model the sensor data in *long* format

The raw matrix is wide (590 columns). At scale you instead store an append-only
event stream — `(wafer_id, sensor_id, reading, event_ts)` — because:

- **It is how telemetry actually arrives** — each sensor emits its own events.
- **Sparsity is free** — a missing reading is simply a row that was never written
  (no 590-wide rows padded with NULLs).
- **A single `GROUP BY sensor_id`** then yields per-sensor statistics across the
  entire history, which is awkward to express across 590 separate columns.

`src/bigdata/melt_to_long.py` performs this reshape (the ETL "landing" step).

## When to reach for Hive vs Pig vs pandas

| Situation | Use | Why |
|---|---|---|
| Data fits in RAM (≤ a few GB), iterative analysis & ML | **pandas / scikit-learn** | Fastest to write, rich ML ecosystem, what this project uses for modelling. |
| Bigger than pandas is comfortable with, but a cluster is overkill; you want fast **SQL with zero infrastructure**, local/single-node | **DuckDB** | Modern in-process columnar OLAP ("SQLite for analytics"). No JVM/server: `pip install duckdb`, query DataFrames/CSV/Parquet directly. The pragmatic middle ground — demonstrated in `notebooks/bigdata_demo_pig_duckdb.ipynb`. |
| Data lives in HDFS / a warehouse; you want **SQL** aggregates, BI tool access, partitioned scans over TB | **Hive** | Declarative SQL over massive tables; integrates with dashboards; great for `GROUP BY`, `JOIN`, windowed KPIs like daily yield. |
| Complex **multi-step ETL** pipelines, semi-structured data, procedural control over each transform | **Pig** | Pig Latin reads as a dataflow of named relations (`LOAD → FILTER → GROUP → FOREACH → STORE`); easier than deeply nested SQL for long cleaning chains. |

Rule of thumb for this project: **prototype and model in pandas; push the
recurring, full-history aggregation down into Hive/Pig** so the warehouse does the
work and the data scientist only pulls a compact, query-ready extract.

## What the scripts compute

Both `secom_ingest.hql` (Hive) and `secom_aggregate.pig` (Pig) produce the same
fab-scale artefacts:

1. **Per-sensor profile** — count, mean, std, min, max for every sensor
   (`GROUP BY sensor_id`).
2. **Near-constant sensor list** — `HAVING std < 1e-6` / `FILTER std > 1e-6`,
   mirroring the variance pruning in `preprocessing.py`.
3. **Missingness per sensor** — `1567 − n_readings` (Hive version).
4. **Pass/fail mean shift** — `JOIN` sensor readings to the wafer label table and
   compare the mean reading for pass vs fail wafers; the largest shifts are
   candidate yield signals — the warehouse-scale echo of the notebook's feature
   importance and box-plot views.
5. **Daily yield KPI** — `GROUP BY run_date` on the timestamp (Hive), the metric a
   fab manager actually watches.

## How to run

### Easiest: the runnable Colab demo

Open **`notebooks/bigdata_demo_pig_hive.ipynb`** (Pig + HiveQL via Spark SQL) or
**`notebooks/bigdata_demo_pig_duckdb.ipynb`** (Pig + DuckDB, zero-Java) and
`Runtime -> Run all`. Each melts SECOM, runs the **real Pig Latin in local mode**
(`pig -x local`), and runs the same aggregations in its SQL engine — printing
result tables you can screenshot. No cluster required. For a genuine warehouse
stack, **`notebooks/bigdata_demo_hadoop_hive.ipynb`** installs real **Hadoop + Hive**
and executes `secom_ingest.hql` for real (heaviest; ~5-10 min).

### On a real cluster / laptop

```bash
# 1. Land the long-format files (local or HDFS)
python -m src.bigdata.melt_to_long

# 2a. Hive
hive -f src/bigdata/secom_ingest.hql

# 2b. Pig — cluster (uses the HDFS default paths baked into the script)
pig -f src/bigdata/secom_aggregate.pig

# 2c. Pig — local smoke test (override the path parameters)
pig -x local \
    -param LONG_INPUT=data/processed/long/secom_sensor_long.tsv \
    -param LABELS_INPUT=data/processed/long/secom_labels.tsv \
    -param OUT_DIR=/tmp/pig_out \
    -f src/bigdata/secom_aggregate.pig
```

> The `LOCATION` paths in the HiveQL (`/warehouse/secom/...`) assume the TSVs were
> copied to HDFS (`hdfs dfs -put`). For a local Hive test, point `LOCATION` at the
> `data/processed/long/` directory instead.
