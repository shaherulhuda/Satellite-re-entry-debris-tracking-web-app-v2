import numpy as np

import debris
import globe
import orbits


def test_fast_orbit_paths_match_the_precise_ones():
    recs = [r for r in orbits.load_records() if r["NORAD_CAT_ID"] == 25544] + debris.load_debris_records()[:3]
    t0 = orbits.TS.now()
    fast = orbits.orbit_paths_fast(recs, n_points=48, start=t0)
    assert len(fast) == len(recs)
    for rec, f in zip(recs, fast):
        ref = orbits.orbit_path_xyz(rec, 48, start=t0)
        err = np.sqrt((ref["x"] - f["x"]) ** 2 + (ref["y"] - f["y"]) ** 2 + (ref["z"] - f["z"]) ** 2)
        assert err.max() < 3.0  # km: sidereal-angle rotation + rounding to whole km
        assert abs(ref["period_s"] - f["period_s"]) < 1e-6


def test_fast_paths_skip_objects_that_cannot_be_propagated():
    bad = dict(debris.load_debris_records()[0], ECCENTRICITY=0.9999, MEAN_MOTION=40.0)
    good = debris.load_debris_records()[1]
    out = orbits.orbit_paths_fast([bad, good], n_points=16)
    assert 1 <= len(out) <= 2 and all(np.isfinite(p["x"]).all() for p in out)


def test_globe_html_has_a_toggle_only_when_debris_is_given():
    recs = orbits.load_records()[:2]
    paths = [dict(orbits.orbit_path_xyz(r, 30), name=r["OBJECT_NAME"]) for r in recs]
    deb = orbits.orbit_paths_fast(debris.load_debris_records()[:20], n_points=24)
    with_deb = globe.orbit_html(paths, debris=deb)
    without = globe.orbit_html(paths)
    assert "Debris orbits" in with_deb and 'id="deb"' in with_deb and '"count":20' in with_deb.replace(" ", "")
    assert '"deb":null' in without.replace(" ", "")
