import numpy as np, pandas as pd
from wq_qaqc import mann_kendall, to_utm, qaqc, make_demo

def test_mk_detects_trend():
    z, p, sen = mann_kendall(np.arange(50) * 0.5 + np.random.default_rng(0).normal(0, .1, 50))
    assert p < 0.001 and abs(sen - 0.5) < 0.05

def test_mk_no_trend():
    _, p, _ = mann_kendall(np.random.default_rng(1).normal(0, 1, 60))
    assert p > 0.05

def test_utm_known_point():
    # Toronto CN Tower approx (43.6426, -79.3871) -> UTM 17N approx E 630,085, N 4,833,445
    x, y = to_utm(pd.DataFrame({"lat": [43.6426], "lon": [-79.3871]}))
    assert abs(x[0] - 630085) < 50 and abs(y[0] - 4833445) < 100

def test_sentinel_flagged():
    q = qaqc(make_demo())
    assert q.loc[q.value == -999, "flag_range"].all()
