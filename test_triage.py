import json
import math

import numpy as np
import pandas as pd
import pytest
from sgp4.api import Satrec

import debris
import decay
import maneuver
import orbits
import triage


@pytest.fixture(scope="module")
def world():
    act, deb = orbits.load_records(), debris.load_debris_records()
    recs = debris.merge_records(act, deb)
    cat = decay.estimate_decay(orbits.catalog_from_records(recs))
    cat["type"] = cat["OBJECT_NAME"].map(decay.classify_type)
    ids = {r["NORAD_CAT_ID"] for r in deb}
    cat["source"] = np.where(cat["NORAD_CAT_ID"].isin(ids), "debris", "active")
    return cat, {r["NORAD_CAT_ID"]: r for r in recs}


# ---- debris file
SAMPLE_TLE = debris.Path(__file__).with_name("testdata") / "fengyun1c_sample.tle"


def test_tle_parser_matches_sgp4_tle_parser():
    text = SAMPLE_TLE.read_text()
    recs, skipped = debris.parse_tle_text(text)
    assert len(recs) == 30 and skipped == 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for k in (0, 15, 29):
        ref = Satrec.twoline2rv(lines[3 * k + 1], lines[3 * k + 2])
        mine = triage.conjunction.to_satrec(recs[k])
        _, r1, _ = ref.sgp4(ref.jdsatepoch, ref.jdsatepochF + 0.25)
        _, r2, _ = mine.sgp4(ref.jdsatepoch, ref.jdsatepochF + 0.25)
        assert np.linalg.norm(np.array(r1) - np.array(r2)) < 1e-3


def test_parser_rejects_garbage_and_merge_prefers_newer():
    recs, skipped = debris.parse_tle_text("NAME\nnot a tle\nalso not\n")
    assert recs == [] and skipped == 1
    a = {"NORAD_CAT_ID": 1, "EPOCH": "2026-01-01T00:00:00"}
    b = {"NORAD_CAT_ID": 1, "EPOCH": "2026-02-01T00:00:00"}
    assert debris.merge_records([a], [b]) == [b] and debris.merge_records([b], [a]) == [b]


# ---- maneuver
def test_shift_per_dv_matches_hand_calculation():
    # 12 h before closest approach at ~15.5 rev/day: about 133 km of shift per 1 m/s
    assert maneuver.shift_per_unit_dv_km(15.5, 12 * 3600) == pytest.approx(133.2, rel=0.02)


def test_avoidance_plan_logic():
    plan = maneuver.avoidance_plan(2.0, 5.0, 15.5, 800, 1500)
    assert (plan["dv_worst_ms"] > plan["dv_best_ms"]).all() and (plan["fuel_worst_kg"] > plan["fuel_best_kg"]).all()
    assert plan["dv_worst_ms"].iloc[0] > plan["dv_worst_ms"].iloc[-1]  # burn earlier -> cheaper
    assert maneuver.avoidance_plan(6.0, 5.0, 15.5, 800, 1500).empty
    dv, isp, m = 0.1, 1500, 800
    assert maneuver.propellant_kg(m, dv, isp) == pytest.approx(m * dv / (maneuver.G0 * isp), rel=1e-4)


# ---- screening
def test_asset_vs_debris_finds_planted_debris_and_ignores_formation(world):
    cat, recs = world
    iss = recs[25544]
    twin = dict(iss, NORAD_CAT_ID=339001, OBJECT_NAME="PLANTED DEB", MEAN_ANOMALY=(iss["MEAN_ANOMALY"] + 0.3) % 360)
    glued = dict(iss, NORAD_CAT_ID=339002, OBJECT_NAME="GLUED DEB", MEAN_ANOMALY=(iss["MEAN_ANOMALY"] + 0.01) % 360)
    extra = orbits.catalog_from_records([twin, glued])
    extra = decay.estimate_decay(extra)
    extra["type"], extra["source"] = "debris", "debris"
    cat2 = pd.concat([cat, extra], ignore_index=True)
    recs2 = {**recs, 339001: twin, 339002: glued}
    s = triage.conjunction.to_satrec(iss)
    ev, n = triage.screen_asset_vs_debris(iss, cat2, recs2, s.jdsatepoch + s.jdsatepochF, hours=2, threshold_km=60)
    assert n >= 2
    names = set(ev["name"])
    assert "PLANTED DEB" in names and "GLUED DEB" not in names
    miss = ev[ev["name"] == "PLANTED DEB"]["miss_km"].min()
    assert 30 < miss < 42  # chord of 0.3 deg at ~6800 km is about 35.6 km
    assert {"other_perigee_km", "other_type"} <= set(ev.columns)


# ---- facts for the LLM are plain JSON
def test_memo_facts_are_json_serialisable(world):
    import llm
    cat, recs = world
    ctx = llm.DataContext(cat, recs, decay.risk_components(cat))
    iss = recs[25544]
    s = triage.conjunction.to_satrec(iss)
    ev, n = triage.screen_asset(iss, cat, recs, s.jdsatepoch + s.jdsatepochF, hours=24, threshold_km=60)
    plan = maneuver.avoidance_plan(3.0, 5.0, 15.5, 420000, 300)
    facts = triage.asset_facts(llm.object_facts(ctx, 25544, include_position=False), ev, n, 24, 60, "t", plan,
                               {"mass_kg": 420000}, None)
    json.dumps(facts)


def test_screen_asset_against_modes(world):
    cat, recs = world
    iss = recs[25544]
    s = triage.conjunction.to_satrec(iss)
    start = s.jdsatepoch + s.jdsatepochF
    deb_ev, n_deb = triage.screen_asset(iss, cat, recs, start, hours=2, threshold_km=100, against="debris")
    act_ev, n_act = triage.screen_asset(iss, cat, recs, start, hours=2, threshold_km=100, against="active")
    all_ev, n_all = triage.screen_asset(iss, cat, recs, start, hours=2, threshold_km=100, against="all")
    assert n_all == n_deb + n_act
    src = dict(zip(cat["NORAD_CAT_ID"], cat["source"]))
    assert all(src[i] == "debris" for i in deb_ev["norad_id"]) and all(src[i] == "active" for i in act_ev["norad_id"])
    assert len(all_ev) == len(deb_ev) + len(act_ev)
    groups = sorted(cat.loc[cat.source == "debris", "GROUP"].dropna().unique())
    _, n_one = triage.screen_asset(iss, cat, recs, start, hours=1, threshold_km=50, against="debris", groups=groups[:1])
    assert n_one <= n_deb
    with pytest.raises(ValueError):
        triage.screen_asset(iss, cat, recs, start, against="nope")


def test_hermite_minimum_matches_straight_line_geometry():
    import conjunction
    # relative motion r(t) = (100 - 10 t, 50, 0) km over a 120 s step: closest approach is 50 km at t = 10 s
    v = np.array([[-10.0, 0.0, 0.0]])
    P0, P1 = np.array([[100.0, 50.0, 0.0]]), np.array([[100.0 - 10 * 120, 50.0, 0.0]])
    u, d = conjunction._hermite_minimum(P0, v * 120, P1, v * 120, np.array([0.0]), np.array([0.5]))
    assert d[0] == pytest.approx(50.0, abs=1e-3) and u[0] * 120 == pytest.approx(10.0, abs=0.01)


def test_screening_window_is_cut_where_the_object_decays(world):
    cat, recs = world
    # the lowest-perigee active satellites propagate only a day or two: find one whose 72 h window gets cut short
    # (which objects those are changes with every data refresh, so none is hard-coded)
    for nid in cat[cat.source == "active"].nsmallest(15, "perigee_km")["NORAD_CAT_ID"]:
        rec = recs[int(nid)]
        s = triage.conjunction.to_satrec(rec)
        start = s.jdsatepoch + s.jdsatepochF
        ev, _ = triage.conjunction.screen(rec, cat, recs, start, hours=72, threshold_km=30)
        if ev.attrs["window_hours_effective"] < 72 - 1:
            assert 0 < ev.attrs["window_hours_effective"] < 72
            with pytest.raises(ValueError, match="cannot be propagated"):
                triage.conjunction.screen(rec, cat, recs, start + 5, hours=24, threshold_km=30)
            return
    pytest.skip("no satellite in this data set decays inside a 72 h window")


def test_long_window_agrees_with_short_windows(world):
    cat, recs = world
    iss = recs[25544]
    s = triage.conjunction.to_satrec(iss)
    start = s.jdsatepoch + s.jdsatepochF
    long_ev, _ = triage.conjunction.screen(iss, cat, recs, start, hours=72, threshold_km=25)
    first, _ = triage.conjunction.screen(iss, cat, recs, start, hours=24, threshold_km=25)
    key = lambda d: {(int(n), round(j * 1440)) for n, j in zip(d.norad_id, d.tca_jd)}
    assert key(first) <= key(long_ev)  # every pass in the first day also shows up in the 3-day screen
    assert len(long_ev) > len(first)
