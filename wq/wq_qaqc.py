"""Water-quality time-series QA/QC and trend analysis.

Usage:
    python wq_qaqc.py                      # runs on synthetic demo data
    python wq_qaqc.py --csv my_data.csv    # columns: date, station, lat, lon, value
                                           #   value = e.g. specific conductance (uS/cm)

Steps: (1) range + unit checks, (2) gap detection, (3) rolling robust-z outlier flags,
(4) Mann-Kendall trend with Sen's slope on QA-passed, monthly-median data
(5) map in a projected CRS (UTM 17N, EPSG:32617) with a QA summary figure.
All assumptions are written to outputs/run_log.json for reproducibility.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
PHYS_MIN, PHYS_MAX = 0.0, 5000.0   # plausible range, uS/cm (assumption: freshwater)

def make_demo(seed=SEED):
    """SYNTHETIC data: 3 stations, daily, seasonal cycle + slow upward trend + injected faults."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    stations = {"S1": (43.65, -79.38), "S2": (43.72, -79.45), "S3": (43.80, -79.30)}
    rows = []
    for sid, (lat, lon) in stations.items():
        t = np.arange(len(dates))
        v = 400 + 0.02 * t + 60 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 15, len(dates))
        v = pd.Series(v, index=dates)
        v.iloc[rng.choice(len(v), 12, replace=False)] += rng.choice([-1, 1], 12) * 300  # spikes
        v.iloc[900:960] = np.nan                                                        # outage
        v.iloc[1500:1510] = -999                                                        # sentinel
        rows.append(pd.DataFrame({"date": dates, "station": sid, "lat": lat, "lon": lon, "value": v.values}))
    return pd.concat(rows, ignore_index=True)

def qaqc(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date"])
    df["flag_range"] = (df["value"] < PHYS_MIN) | (df["value"] > PHYS_MAX)
    df["flag_missing"] = df["value"].isna()
    clean = df["value"].where(~(df["flag_range"] | df["flag_missing"]))
    g = clean.groupby(df["station"])
    med = g.transform(lambda s: s.rolling(31, center=True, min_periods=10).median())
    mad = (clean - med).abs().groupby(df["station"]).transform(
        lambda s: s.rolling(31, center=True, min_periods=10).median())
    rz = 0.6745 * (clean - med) / mad.replace(0, np.nan)   # robust z-score
    df["flag_outlier"] = rz.abs() > 3.5
    df["qa_pass"] = ~(df["flag_range"] | df["flag_missing"] | df["flag_outlier"])
    return df

def gaps(df, min_days=7):
    out = []
    for sid, g in df[df["qa_pass"]].groupby("station"):
        d = g["date"].diff().dt.days
        for i in d[d > min_days].index:
            out.append({"station": sid, "gap_days": int(d[i]), "ends": str(g.loc[i, "date"].date())})
    return out

def mann_kendall(y):
    """Mann-Kendall (no tie/autocorr correction) + Sen's slope per step."""
    y = np.asarray(y, float); n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = 0 if s == 0 else (s - np.sign(s)) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return z, p, float(np.median(slopes))

def trends(df):
    res = {}
    for sid, g in df[df["qa_pass"]].groupby("station"):
        m = g.set_index("date")["value"].resample("MS").median().dropna()
        # deseasonalise: subtract monthly climatology so seasonality doesn't dominate
        m = m - m.groupby(m.index.month).transform("mean")
        z, p, sen = mann_kendall(m.values)
        res[sid] = {"n_months": int(len(m)), "z": round(float(z), 2), "p": float(p),
                    "sen_slope_per_year": round(sen * 12, 2)}
    return res

def to_utm(df):
    """Lon/lat (EPSG:4326) -> UTM 17N (EPSG:32617) without needing pyproj (Transverse Mercator)."""
    a, f = 6378137.0, 1 / 298.257223563
    k0, lon0 = 0.9996, np.radians(-81.0)
    e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    phi, lam = np.radians(df["lat"].values), np.radians(df["lon"].values)
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T, C, A = np.tan(phi) ** 2, ep2 * np.cos(phi) ** 2, (lam - lon0) * np.cos(phi)
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*phi)
             + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*phi) - (35*e2**3/3072) * np.sin(6*phi))
    x = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120) + 500000
    y = k0 * (M + N * np.tan(phi) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24
              + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720))
    return x, y

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    df = pd.read_csv(a.csv) if a.csv else make_demo()
    q = qaqc(df)
    summ = q.groupby("station")[["flag_range", "flag_missing", "flag_outlier", "qa_pass"]].sum().astype(int)
    tr = trends(q)
    log = {"source": a.csv or "SYNTHETIC demo (seed=42)", "phys_range": [PHYS_MIN, PHYS_MAX],
           "outlier_rule": "rolling 31-day robust z (MAD) > 3.5", "trend_method":
           "Mann-Kendall on deseasonalised monthly medians, Sen slope; no autocorrelation correction (limitation)",
           "crs": "inputs EPSG:4326; map in EPSG:32617 (UTM 17N)", "flag_counts": summ.to_dict("index"),
           "gaps_gt7d": gaps(q), "trends": tr}
    json.dump(log, open("outputs/run_log.json", "w"), indent=2, default=str)
    print(summ); print(json.dumps(tr, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    s1 = q[q.station == "S1"]
    ax[0].plot(s1.date, s1.value.where(s1.value > -100), lw=.4, c="grey", label="raw")
    ax[0].scatter(s1.date[s1.flag_outlier], s1.value[s1.flag_outlier], c="r", s=14, label="outlier flag")
    ax[0].set(title="Station S1: raw series and flagged outliers", ylabel="Conductance (uS/cm)"); ax[0].legend()
    st = q.groupby("station").first()[["lat", "lon"]].reset_index()
    x, y = to_utm(st)
    ax[1].scatter(x, y, s=80)
    for xi, yi, n in zip(x, y, st.station):
        ax[1].annotate(f"{n}\n{tr[n]['sen_slope_per_year']:+.1f}/yr", (xi, yi), textcoords="offset points", xytext=(6, 6))
    ax[1].set(title="Stations (UTM 17N, m) with Sen slope", xlabel="Easting (m)", ylabel="Northing (m)")
    plt.tight_layout(); plt.savefig("outputs/qaqc_summary.png", dpi=130)

if __name__ == "__main__":
    main()
