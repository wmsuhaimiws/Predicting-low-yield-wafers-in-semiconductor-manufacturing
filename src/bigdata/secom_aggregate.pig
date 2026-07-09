/* ===========================================================================
 * secom_aggregate.pig  —  Apache Pig (Pig Latin) fab-scale aggregation
 * ---------------------------------------------------------------------------
 * Application of Tools: the fab-scale aggregation expressed as a Pig
 * dataflow. Pig shines for multi-step ETL where you want to read the
 * transformation as a pipeline of relations rather than nested SQL.
 *
 * Paths are PARAMETERS with cluster defaults, so the SAME script runs:
 *   - on a cluster:   pig -f src/bigdata/secom_aggregate.pig
 *   - locally (demo): pig -x local \
 *                         -param LONG_INPUT=/content/secom_sensor_long.tsv \
 *                         -param LABELS_INPUT=/content/secom_labels.tsv \
 *                         -param OUT_DIR=/content/pig_out \
 *                         -f src/bigdata/secom_aggregate.pig
 * A runnable, end-to-end version is in notebooks/bigdata_demo.ipynb.
 *
 * Inputs (produced by src/bigdata/melt_to_long.py):
 *   LONG_INPUT   : (wafer_id, sensor_id, reading)
 *   LABELS_INPUT : (wafer_id, label_raw, fail, event_ts)
 * =========================================================================== */

%default LONG_INPUT   '/warehouse/secom/sensor_long/secom_sensor_long.tsv'
%default LABELS_INPUT '/warehouse/secom/labels/secom_labels.tsv'
%default OUT_DIR      '/warehouse/secom/out'

-- 1. LOAD the long sensor-event stream and the wafer labels ------------------
sensors = LOAD '$LONG_INPUT'
    USING PigStorage('\t')
    AS (wafer_id:int, sensor_id:int, reading:double);

labels = LOAD '$LABELS_INPUT'
    USING PigStorage('\t')
    AS (wafer_id:int, label_raw:int, fail:int, event_ts:chararray);

-- 2. FILTER out null readings (defensive) ------------------------------------
clean_sensors = FILTER sensors BY reading IS NOT NULL;

-- 3. GROUP BY sensor_id and FOREACH ... GENERATE the per-sensor profile ------
--    Variance is computed as E[x^2] - E[x]^2 so we need NO external UDF
--    (Pig core has no VAR builtin; this keeps the script dependency-free).
squared = FOREACH clean_sensors GENERATE
    sensor_id,
    reading,
    (reading * reading) AS r2;

by_sensor = GROUP squared BY sensor_id;
sensor_prof = FOREACH by_sensor GENERATE
    group                                                   AS sensor_id,
    COUNT(squared)                                          AS n_readings,
    AVG(squared.reading)                                    AS mean_reading,
    (AVG(squared.r2) - AVG(squared.reading) * AVG(squared.reading)) AS variance_reading,
    MIN(squared.reading)                                    AS min_reading,
    MAX(squared.reading)                                    AS max_reading;

-- 4. FILTER to drop near-constant sensors (variance ~ 0) ---------------------
informative = FILTER sensor_prof BY variance_reading > 1e-9;

-- 5. JOIN sensor readings to wafer labels, then split pass vs fail -----------
joined = JOIN clean_sensors BY wafer_id, labels BY wafer_id;

labelled = FOREACH joined GENERATE
    clean_sensors::sensor_id AS sensor_id,
    clean_sensors::reading   AS reading,
    labels::fail             AS fail;

by_sensor_class = GROUP labelled BY sensor_id;
passfail = FOREACH by_sensor_class {
    pass_rows = FILTER labelled BY fail == 0;
    fail_rows = FILTER labelled BY fail == 1;
    GENERATE
        group                  AS sensor_id,
        AVG(pass_rows.reading) AS mean_pass,
        AVG(fail_rows.reading) AS mean_fail;
}

-- 6. Pass/fail mean shift; rank by absolute shift (projected, not in ORDER) --
shift = FOREACH passfail GENERATE
    sensor_id,
    mean_pass,
    mean_fail,
    (mean_fail - mean_pass)      AS mean_shift,
    ABS(mean_fail - mean_pass)   AS abs_shift;

ranked = ORDER shift BY abs_shift DESC;
top15  = LIMIT ranked 15;

-- 7. STORE outputs and echo the strongest pass/fail signals ------------------
STORE informative INTO '$OUT_DIR/sensor_profile' USING PigStorage('\t');
STORE top15        INTO '$OUT_DIR/top_signals'    USING PigStorage('\t');

DUMP top15;
