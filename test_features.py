import numpy as np
import pandas as pd

import conjunction
import decay
import footprint
import history
import orbits


def catalogue():
    return decay.estimate_decay(orbits.load_catalog())


# ---- decay
def test_decay_lower_is_sooner_and_stable_is_none():
    df = catalogue()
    est = df[df["decay_status"] == "estimate"]
    low = est.nsmallest(20, "perigee_km")["decay_days"].median()
    high = est[(est.perigee_km > 450) & (est.perigee_km < 600)]["decay_days"].median()
    assert low < 5 < high
    assert (df[df["decay_status"] == "no measurable decay"]["decay_days"].isna()).all()


def test_decay_integration_matches_closed_form_for_single_scale_height():
    h0, rate = 340.0, 1.0
    days = decay.days_to_reentry(h0, rate)
    assert 0 < days < 1e4
    # rate x 2 -> half the time
    assert abs(decay.days_to_reentry(h0, 2 * rate) * 2 - days) / days < 1e-6


def test_projection_ends_at_reentry_time():
    h0, rate = 400.0, 0.5
    t, h = decay.altitude_projection(h0, rate)
    assert abs(t[-1] - decay.days_to_reentry(h0, rate)) / t[-1] < 1e-3
    assert h[0] == h0 and h[-1] == decay.REENTRY_ALT_KM and np.all(np.diff(t) >= 0)


def test_density_decreases_with_altitude():
    h = np.array([150.0, 300.0, 500.0, 800.0])
    assert np.all(np.diff(decay.density(h)) < 0)


# ---- risk
def test_risk_score_monotonic_and_weights():
    df = catalogue()
    comp = decay.risk_components(df)
    assert ((comp >= 0) & (comp <= 1)).all().all()
    s = decay.risk_score(comp)
    assert s.between(0, 100).all()
    top = df.loc[s.idxmax()]
    assert top["perigee_km"] < 300
    only_type = decay.risk_score(comp, {"perigee": 0, "decay": 0, "inclination": 0, "type": 1})
    assert set(only_type.round(2)) <= {30.0, 80.0, 100.0}


def test_classify_type():
    assert decay.classify_type("COSMOS 2251 DEB") == "debris"
    assert decay.classify_type("SL-4 R/B") == "rocket body"
    assert decay.classify_type("ISS (ZARYA)") == "payload"


# ---- footprint
def test_footprint_geometry():
    assert footprint.max_latitude(51.6) == 51.6
    assert abs(footprint.max_latitude(98.0) - 82.0) < 1e-9
    assert abs(footprint.surface_fraction(90.0) - 1.0) < 1e-9
    lat, p = footprint.latitude_distribution(51.6)
    assert abs(p.sum() - 1) < 1e-9
    assert p[np.abs(lat) > 52].sum() < 1e-9
    assert abs(lat[p.argmax()]) > 45  # time piles up near the band edges


# ---- history
def test_history_dedupes_and_trend(tmp_path):
    df = orbits.load_catalog()
    path = tmp_path / "h.csv"
    n1 = history.append_snapshot(df, path)
    assert n1 > 0 and history.append_snapshot(df, path) == 0
    low = df[df["perigee_km"] <= history.DEFAULT_MAX_PERIGEE_KM].iloc[0]
    later = df.copy()
    later["EPOCH"] = (pd.to_datetime(later["EPOCH"]) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    later["perigee_km"] = later["perigee_km"] - 10
    assert history.append_snapshot(later, path) > 0
    h = history.object_history(history.load_history(path), int(low["NORAD_CAT_ID"]))
    assert len(h) == 2
    assert abs(history.observed_decay_rate(h) - (-2.0)) < 1e-6


# ---- conjunction
def test_conjunction_finds_known_separation_and_ignores_formation():
    df = orbits.load_catalog()
    recs = {r["NORAD_CAT_ID"]: r for r in orbits.load_records()}
    primary = dict(recs[25544])
    # a copy of the primary displaced 0.5 deg along-track (~60 km), same orbit
    twin = dict(primary, NORAD_CAT_ID=339001, OBJECT_NAME="TWIN",
                MEAN_ANOMALY=(primary["MEAN_ANOMALY"] + 0.5) % 360)
    # a copy displaced by ~1 km: a formation flyer, must be ignored
    formation = dict(primary, NORAD_CAT_ID=339002, OBJECT_NAME="FORMATION",
                     MEAN_ANOMALY=(primary["MEAN_ANOMALY"] + 0.01) % 360)
    recs2 = {**recs, 339001: twin, 339002: formation}
    extra = orbits.catalog_from_records([twin, formation])
    cat = pd.concat([df, extra], ignore_index=True)
    s = conjunction.to_satrec(primary)
    ev, n = conjunction.screen(primary, cat, recs2, s.jdsatepoch + s.jdsatepochF, hours=1, threshold_km=100)
    assert n > 0
    names = set(ev["name"])
    assert "TWIN" in names and "FORMATION" not in names
    miss = ev[ev["name"] == "TWIN"]["miss_km"].min()
    assert 50 < miss < 70
