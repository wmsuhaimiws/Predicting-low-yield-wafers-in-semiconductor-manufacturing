-- ===========================================================================
-- secom_ingest.hql  —  Apache Hive ingest + aggregation for SECOM at fab scale
-- ---------------------------------------------------------------------------
-- Application of Tools: expresses the ingest + aggregation step of
-- the data-management pipeline in HiveQL. Run order:
--   1) src/bigdata/melt_to_long.py   -> lands long-format TSVs in HDFS/local
--   2) hive -f src/bigdata/secom_ingest.hql
--
-- Why long format + Hive: a fab streams billions of (wafer, sensor, reading)
-- events per day. They do not fit in pandas memory, but a GROUP BY sensor_id in
-- Hive runs over the whole partitioned warehouse. See docs/bigdata_hive_pig.md.
-- ===========================================================================

CREATE DATABASE IF NOT EXISTS secom_fab
  COMMENT 'Semiconductor wafer-fab process telemetry (UCI SECOM, id 179)';
USE secom_fab;

-- ---------------------------------------------------------------------------
-- 1. EXTERNAL TABLES over the landed files (data stays in place on HDFS)
-- ---------------------------------------------------------------------------

-- 1a. Wafer-level label + timestamp (one row per wafer)
DROP TABLE IF EXISTS secom_labels;
CREATE EXTERNAL TABLE secom_labels (
    wafer_id   INT,
    label_raw  INT,         -- -1 = pass, +1 = fail (original encoding)
    fail        INT,        -- 1 = fail, 0 = pass (modelling target)
    event_ts   STRING       -- acquisition timestamp
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '/warehouse/secom/labels/';     -- hdfs path where melt_to_long.py landed labels

-- 1b. Long sensor-event stream (one row per wafer x sensor reading)
DROP TABLE IF EXISTS secom_sensor_long;
CREATE EXTERNAL TABLE secom_sensor_long (
    wafer_id   INT,
    sensor_id  INT,
    reading    DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '/warehouse/secom/sensor_long/';

-- ---------------------------------------------------------------------------
-- 2. PER-SENSOR PROFILE  (GROUP BY = the core aggregation)
--    Counts, central tendency and spread for every sensor across all wafers.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS sensor_profile;
CREATE TABLE sensor_profile AS
SELECT
    sensor_id,
    COUNT(*)                       AS n_readings,
    ROUND(AVG(reading), 4)         AS mean_reading,
    ROUND(STDDEV_POP(reading), 6)  AS std_reading,
    MIN(reading)                   AS min_reading,
    MAX(reading)                   AS max_reading
FROM secom_sensor_long
GROUP BY sensor_id;

-- 2b. Flag near-constant sensors for pruning (mirrors preprocessing.py).
--     HAVING filters the aggregated groups, not the raw rows.
DROP TABLE IF EXISTS near_constant_sensors;
CREATE TABLE near_constant_sensors AS
SELECT sensor_id, std_reading
FROM sensor_profile
WHERE std_reading < 1e-6              -- effectively zero variance
   OR std_reading IS NULL;

-- ---------------------------------------------------------------------------
-- 3. MISSINGNESS per sensor (expected 1,567 readings per sensor if complete)
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS sensor_missingness;
CREATE TABLE sensor_missingness AS
SELECT
    p.sensor_id,
    p.n_readings,
    (1567 - p.n_readings)                       AS n_missing,
    ROUND((1567 - p.n_readings) / 1567.0, 4)    AS missing_frac
FROM sensor_profile p
ORDER BY missing_frac DESC;

-- ---------------------------------------------------------------------------
-- 4. JOIN sensors to labels: mean reading split by pass vs fail.
--    This is the fab-scale version of the pass/fail comparison the notebook
--    plots — but computed over the entire warehouse, not a sample.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS sensor_passfail_means;
CREATE TABLE sensor_passfail_means AS
SELECT
    s.sensor_id,
    AVG(CASE WHEN l.fail = 0 THEN s.reading END) AS mean_pass,
    AVG(CASE WHEN l.fail = 1 THEN s.reading END) AS mean_fail,
    AVG(CASE WHEN l.fail = 1 THEN s.reading END)
      - AVG(CASE WHEN l.fail = 0 THEN s.reading END) AS mean_shift
FROM secom_sensor_long s
JOIN secom_labels l
  ON s.wafer_id = l.wafer_id
GROUP BY s.sensor_id;

-- 4b. Sensors whose mean shifts most between pass and fail = candidate signals.
SELECT sensor_id, mean_pass, mean_fail, mean_shift
FROM sensor_passfail_means
WHERE mean_shift IS NOT NULL
ORDER BY ABS(mean_shift) DESC
LIMIT 15;

-- ---------------------------------------------------------------------------
-- 5. Daily fab yield (uses the timestamp) — a management-level KPI.
-- ---------------------------------------------------------------------------
SELECT
    SUBSTR(event_ts, 1, 10)                       AS run_date,
    COUNT(*)                                       AS wafers,
    SUM(fail)                                      AS fails,
    ROUND(1 - SUM(fail) / COUNT(*), 4)             AS yield_rate
FROM secom_labels
GROUP BY SUBSTR(event_ts, 1, 10)
ORDER BY run_date;
