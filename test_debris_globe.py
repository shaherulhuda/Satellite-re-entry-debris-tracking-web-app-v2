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


def test_globe_conjunction_markers_and_cyan_tier():
    import json
    import re
    import numpy as np
    import globe
    path = {"x": np.array([7000.0, 0, -7000, 0]), "y": np.array([0, 7000.0, 0, -7000]), "z": np.zeros(4), "period_s": 5400.0,
            "name": "A", "tier": 3}
    html = globe.orbit_html([path], events=[{"x": 7000.0, "y": 0.0, "z": 0.0, "tier": 0, "hover": "B<br>1.0 km"}])
    ev = json.loads(re.search(r'"ev":(\{.*?\})', html).group(1))
    assert ev["x"] == [7000.0] and ev["idx"] >= 2
    assert "#4fd1e0" in html and "1.0 km" in html
    assert '"ev":null' in globe.orbit_html([path])


def test_globe_can_start_paused():
    import numpy as np
    import globe
    path = {"x": np.array([7000.0, 0, -7000, 0]), "y": np.array([0, 7000.0, 0, -7000]), "z": np.zeros(4), "period_s": 5400.0, "name": "A"}
    assert '"autoplay":false' in globe.orbit_html([path])  # paused by default
    assert '"autoplay":true' in globe.orbit_html([path], autoplay=True)
    assert "speed = 60," in globe.orbit_html([path]) and '<option value="60" selected>' in globe.orbit_html([path])


def test_ground_track_map_fills_the_width():
    import orbits
    import globe
    rec = orbits.load_records()[0]
    fig = globe.ground_track_figure(orbits.ground_track(rec), orbits.current_position(rec))
    assert fig.layout.yaxis.scaleanchor is None and fig.layout.height >= 440


def test_globe_and_charts_are_transparent():
    import numpy as np
    import charts
    import globe
    path = {"x": np.array([7000.0, 0, -7000, 0]), "y": np.array([0, 7000.0, 0, -7000]), "z": np.zeros(4), "period_s": 5400.0, "name": "A"}
    html = globe.orbit_html([path])
    assert "html,body{background:transparent" in html and "color-scheme" not in html.split("</head>")[0]
    for fig in (charts.decay_figure(400.0, 0.5), charts.footprint_figure(51.6)):
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)" and fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
        assert fig.layout.xaxis.gridcolor == globe.GRID_COLOR
