import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset
import decay
import globe
import orbits


def frame(miss, speed, hours):
    return pd.DataFrame({"perigee_km": [200.0] * len(miss), "decay_days": [np.nan] * len(miss),
                         "INCLINATION": [50.0] * len(miss), "OBJECT_NAME": ["X"] * len(miss),
                         "ca_miss_km": miss, "ca_speed_km_s": speed, "ca_hours": hours})


def test_close_approach_components():
    c = decay.risk_components(frame([1.0, 5.0, 25.0, 60.0, np.nan], [15.0, 7.5, 12.0, 12.0, np.nan], [0.0, 15.0, 3.0, 3.0, np.nan]))
    assert c.loc[0, ["s_miss", "s_speed", "s_time"]].tolist() == [1.0, 1.0, 1.0]
    assert c.loc[1, "s_miss"] == pytest.approx(0.5, abs=0.01) and c.loc[1, "s_speed"] == pytest.approx(0.5) \
        and c.loc[1, "s_time"] == pytest.approx(0.5)
    assert c.loc[2:, ["s_miss", "s_speed", "s_time"]].to_numpy().sum() == 0  # too far, or no approach


def test_missing_close_approach_columns_count_as_no_approach():
    df = frame([1.0], [15.0], [0.0]).drop(columns=["ca_miss_km", "ca_speed_km_s", "ca_hours"])
    assert decay.risk_components(df)[["s_miss", "s_speed", "s_time"]].to_numpy().sum() == 0


def test_default_weights_sum_to_one_and_each_term_moves_the_score():
    assert sum(decay.DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    c = decay.risk_components(frame([1.0, 30.0], [15.0, 15.0], [0.0, 0.0]))
    for key in ("miss", "speed", "time"):
        only = decay.risk_score(c, {k: (1.0 if k == key else 0.0) for k in decay.DEFAULT_WEIGHTS})
        assert only.iloc[0] > only.iloc[1] == 0


def test_traffic_light_is_by_overall_rank():
    s = pd.Series(np.linspace(100, 1, 200))
    lv = decay.traffic_light(s)
    assert lv.iloc[:10].eq("red").all() and lv.iloc[10:40].eq("yellow").all() and lv.iloc[40:].eq("green").all()
    # listing the top 50 gives 10 red / 30 yellow / 10 green; listing the top 10 gives all red
    assert lv.iloc[:50].value_counts().to_dict() == {"yellow": 30, "red": 10, "green": 10}
    assert lv.iloc[:10].eq("red").all()
    shuffled = s.sample(frac=1, random_state=1)
    assert decay.traffic_light(shuffled).value_counts().to_dict() == {"green": 160, "yellow": 30, "red": 10}
    assert len(decay.traffic_light(pd.Series([5.0]))) == 1


def test_attach_closest_approaches():
    cat, _ = dataset.load_catalogue()
    ids = [int(cat.NORAD_CAT_ID.iloc[0]), int(cat.NORAD_CAT_ID.iloc[1])]
    ca = {"rows": [{"norad_id": ids[0], "miss_km": 3.2, "rel_speed_km_s": 9.0, "hours_to_ca": 5.0, "with_name": "A"},
                   {"norad_id": ids[1], "miss_km": None, "rel_speed_km_s": None, "hours_to_ca": None, "with_name": None}]}
    out = dataset.attach_closest_approaches(cat, ca)
    assert out["ca_screened"].sum() == 2
    r = out.set_index("NORAD_CAT_ID").loc[ids[0]]
    assert r.ca_miss_km == 3.2 and r.ca_with == "A" and r.ca_hours == 5.0
    assert pd.isna(out.set_index("NORAD_CAT_ID").loc[ids[1]].ca_miss_km)
    assert not dataset.attach_closest_approaches(cat, None)["ca_screened"].any()


def test_compute_script_runs_on_a_tiny_shortlist():
    spec = importlib.util.spec_from_file_location("compute_conjunctions", Path(__file__).parent.parent / "tools" / "compute_conjunctions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_conjunctions"] = mod  # worker processes unpickle the task function by module name
    spec.loader.exec_module(mod)
    out = mod.compute(top=3, hours=1.0, workers=1, log=lambda *_: None)
    assert len(out["rows"]) == 3 and out["window_hours"] == 1.0 and out["start_utc"]
    for r in out["rows"]:
        assert {"norad_id", "miss_km", "rel_speed_km_s", "hours_to_ca", "with_name"} <= set(r)
        if r["miss_km"] is not None:
            assert 0 <= r["hours_to_ca"] <= 1.01 and r["rel_speed_km_s"] > 0


def test_globe_colours_by_tier():
    recs = orbits.load_records()[:3]
    paths = [dict(orbits.orbit_path_xyz(r, 20), name=r["OBJECT_NAME"], tier=i) for i, r in enumerate(recs)]
    html = globe.orbit_html(paths)
    assert all(c in html for c in globe.TIER_COLORS)
    plain = globe.orbit_html([dict(p, tier=None) for p in paths[:1]] if False else [{k: v for k, v in paths[0].items() if k != "tier"}])
    assert "#e5484d" not in plain
