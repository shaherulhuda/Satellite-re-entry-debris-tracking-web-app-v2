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
