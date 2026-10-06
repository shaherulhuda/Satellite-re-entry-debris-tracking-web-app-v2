import orbits


def test_catalog_and_watchlist():
    df = orbits.load_catalog()
    assert len(df) > 1000
    wl = orbits.watchlist(df)
    assert len(wl) == 50
    assert wl["Perigee (km)"].is_monotonic_increasing
    assert (df["apogee_km"] >= df["perigee_km"]).all()
    assert orbits.snapshot_date(df).startswith("20")


def test_tracker():
    df = orbits.load_catalog()
    rec = orbits.load_records()[0]
    pos = orbits.current_position(rec)
    assert -90 <= pos["lat"] <= 90 and -180 <= pos["lon"] <= 180
    assert len(orbits.ground_track(rec)) > 100


def test_orbit_path_and_globe_html():
    import globe
    rec = orbits.load_records()[0]
    p = orbits.orbit_path_xyz(rec)
    p["name"] = rec["OBJECT_NAME"]
    r = (p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2) ** 0.5
    assert r.min() > orbits.EARTH_RADIUS_KM
    html = globe.orbit_html([p])
    assert "requestAnimationFrame" in html and "Pause" in html
    assert "https://" not in html.split("<script", 1)[0]  # nothing fetched at load


def test_path_is_earth_fixed_at_start():
    import math
    rec = orbits.load_records()[0]
    t0 = orbits.TS.utc(2026, 10, 6, 12, 0, 0)
    p = orbits.orbit_path_xyz(rec, start=t0)
    lon = math.degrees(math.atan2(p["y"][0], p["x"][0]))
    ref = orbits.current_position(rec, when=t0)["lon"]
    assert abs((lon - ref + 180) % 360 - 180) < 0.05
