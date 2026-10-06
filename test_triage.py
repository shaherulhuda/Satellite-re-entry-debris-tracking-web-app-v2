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
def test_bundled_debris_file_parses_fully_and_matches_sgp4_tle_parser():
    text = debris.DEBRIS_PATH.read_text()
    recs, skipped = debris.parse_tle_text(text)
    assert len(recs) == 1983 and skipped == 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for k in (0, 700, 1982):
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


def test_debris_vs_satellites_only_uses_non_debris(world):
    cat, recs = world
    d = int(cat[(cat.source == "debris") & cat.decay_days.notna()].nsmallest(1, "decay_days").iloc[0]["NORAD_CAT_ID"])
    s = triage.conjunction.to_satrec(recs[d])
    ev, n = triage.screen_debris_vs_satellites(recs[d], cat, recs, s.jdsatepoch + s.jdsatepochF, hours=1, threshold_km=50)
    assert n > 0 and not set(ev["norad_id"]) & set(cat[cat.source == "debris"]["NORAD_CAT_ID"])


# ---- bands and crossings
def test_altitude_bands_find_iss_and_a_dominant_constellation(world):
    bands = triage.altitude_bands(world[0])
    labels = [b["label"] for b in bands]
    assert "ISS orbit" in labels and any(l.endswith("shell") for l in labels)
    assert all(b["lo"] < b["hi"] for b in bands)


def test_band_crossings_ordering_and_inside_flag():
    bands = [{"lo": 300.0, "hi": 400.0, "n_objects": 10, "label": "low"},
             {"lo": 480.0, "hi": 560.0, "n_objects": 20, "label": "mid"},
             {"lo": 900.0, "hi": 950.0, "n_objects": 5, "label": "above"}]
    cr = triage.band_crossings(520.0, 0.5, bands)
    assert list(cr["band"].str.split(" ").str[0]) == ["mid", "low"]  # 'above' skipped
    assert bool(cr.iloc[0]["already_inside"]) and cr.iloc[0]["enters_days"] == 0
    assert cr.iloc[1]["enters_days"] > 0 and cr.iloc[1]["leaves_days"] > cr.iloc[1]["enters_days"]
    assert triage.band_crossings(520.0, float("nan"), bands).empty


# ---- facts for the LLM are plain JSON
def test_memo_facts_are_json_serialisable(world):
    import llm
    cat, recs = world
    ctx = llm.DataContext(cat, recs, decay.risk_components(cat))
    iss = recs[25544]
    s = triage.conjunction.to_satrec(iss)
    ev, n = triage.screen_asset_vs_debris(iss, cat, recs, s.jdsatepoch + s.jdsatepochF, hours=24, threshold_km=60)
    plan = maneuver.avoidance_plan(3.0, 5.0, 15.5, 420000, 300)
    facts = triage.asset_facts(llm.object_facts(ctx, 25544, include_position=False), ev, n, 24, 60, "t", plan,
                               {"mass_kg": 420000}, None)
    json.dumps(facts)
    d = int(cat[(cat.source == "debris") & cat.decay_days.notna()].nsmallest(1, "decay_days").iloc[0]["NORAD_CAT_ID"])
    row = cat[cat["NORAD_CAT_ID"] == d].iloc[0]
    cr = triage.band_crossings(float(decay.mean_altitude_km(row.MEAN_MOTION)), float(row.decay_km_day), triage.altitude_bands(cat))
    json.dumps(triage.eol_facts(llm.object_facts(ctx, d, include_position=False), cr, ev, n, 24, 60, "t", triage.altitude_bands(cat)))
