# Water-Quality Time-Series QA/QC and Trend Analysis

Reproducible Python workflow for screening monitoring data and testing for long-term trends.

**Data:** the default run uses **synthetic** data (3 stations, daily, 2018-2023, seasonal cycle + known upward trend of about 7.3 uS/cm per year, plus injected spikes, an outage and -999 sentinel values) so the pipeline can be checked against a known answer. To use real data (e.g. Ontario PWQMN or ECCC/WSC downloads), run `python wq_qaqc.py --csv yourfile.csv` with columns `date, station, lat, lon, value`.

## Workflow
1. **Range/unit check:** physical bounds and sentinel values (-999) are flagged.
2. **Missing data and gap report:** gaps over 7 days listed per station.
3. **Outliers:** rolling 31-day robust z-score (median/MAD) greater than 3.5.
4. **Trend:** Mann-Kendall test and Sen's slope on deseasonalised monthly medians of QA-passed data.
5. **Mapping:** station coordinates converted from EPSG:4326 to UTM 17N (EPSG:32617); checked against a known point in tests.
6. **Run log:** all assumptions and results saved to `outputs/run_log.json`.

## Validation
The recovered Sen slopes (6.9-7.5 per year) match the injected trend (7.3 per year). `pytest` checks the trend test on known trend / no-trend series, the UTM conversion against a known location, and sentinel flagging.

## Limitations
- Mann-Kendall here has no autocorrelation or tie correction, so p-values may be too optimistic.
- Thresholds (range, z = 3.5) are assumptions for freshwater conductance, not regulatory limits.
- Outlier flags are screening aids, not deletions; review before removing data.

## Run
    pip install -r requirements.txt
    python wq_qaqc.py
    pytest
