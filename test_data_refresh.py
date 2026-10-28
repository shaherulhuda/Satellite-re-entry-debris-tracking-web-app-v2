import datetime as dt
import json

import pandas as pd
import pytest

import debris
import orbits
import snapshot



def load_refresh():
    import importlib.util
    spec = importlib.util.spec_from_file_location("refresh_data", debris.Path(__file__).with_name("tools") / "refresh_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def omm(norad, epoch, name="X DEB", mm=14.0):
    return {"OBJECT_NAME": name, "OBJECT_ID": "2000-001A", "EPOCH": epoch, "MEAN_MOTION": mm, "ECCENTRICITY": 0.001,
            "INCLINATION": 90.0, "RA_OF_ASC_NODE": 10.0, "ARG_OF_PERICENTER": 20.0, "MEAN_ANOMALY": 30.0,
            "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": norad, "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 1, "BSTAR": 1e-4, "MEAN_MOTION_DOT": 1e-5, "MEAN_MOTION_DDOT": 0.0}


# ---- loader
def test_loader_reads_tle_and_json_groups_and_dedupes(tmp_path):
    (tmp_path / "fengyun-1c.tle").write_text(debris.DEBRIS_PATH.read_text())
    json.dump([omm(900001, "2026-10-20T00:00:00", "IRIDIUM 33 DEB"), omm(900002, "2026-10-20T00:00:00", "IRIDIUM 33 DEB")],
              open(tmp_path / "iridium-33.json", "w"))
    json.dump([omm(900002, "2026-10-21T00:00:00", "NEWER COPY"), omm(900003, "2026-10-21T00:00:00", "COSMOS 2251 DEB"),
               {"bad": "record"}], open(tmp_path / "cosmos-2251.json", "w"))
    (tmp_path / "notes.md").write_text("ignored")
    recs = debris.load_debris_records(tmp_path)
    by = {r["NORAD_CAT_ID"]: r for r in recs}
    assert len(recs) == 1983 + 3
    assert by[900001]["GROUP"] == "Iridium-33" and by[900003]["GROUP"] == "Cosmos-2251"
    assert by[900002]["OBJECT_NAME"] == "NEWER COPY"  # newer epoch wins across groups
    assert debris.load_debris_records(tmp_path / "missing") == []


# ---- refresh script
def fake_fetch_factory(n_active=6000, epoch="2026-10-27T12:00:00"):
    calls = []

    def fetch(url):
        calls.append(url)
        if "GROUP=active" in url:
            return json.dumps([omm(100000 + i, epoch, f"SAT {i}") for i in range(n_active)])
        return json.dumps([omm(500000 + len(calls) * 10 + i, epoch, "DEB") for i in range(5)])
    return fetch, calls


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point the refresh script at a temporary copy of the repo's data locations."""
    mod = load_refresh()
    (tmp_path / "data" / "debris").mkdir(parents=True)
    (tmp_path / "data" / "debris" / "fengyun-1c.tle").write_text("old stale file")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(debris, "DEBRIS_DIR", tmp_path / "data" / "debris")
    monkeypatch.setattr(snapshot, "MANIFEST_PATH", tmp_path / "data" / "snapshot.json")
    monkeypatch.setattr(mod.history, "append_snapshot", lambda *a, **k: 0)
    monkeypatch.setattr(mod.orbits, "load_catalog", lambda *a, **k: None)
    return mod, tmp_path


def test_refresh_downloads_everything_together_and_writes_manifest(sandbox):
    mod, root = sandbox
    fetch, calls = fake_fetch_factory()
    now = dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc)
    m = mod.refresh(fetch=fetch, now=now)
    assert len(calls) == 4 and any("1999-025" in c for c in calls) and any("iridium-33-debris" in c for c in calls) \
        and any("cosmos-2251-debris" in c for c in calls)
    assert (root / "active.json").exists() and (root / "data" / "debris" / "cosmos-2251.json").exists()
    assert not (root / "data" / "debris" / "fengyun-1c.tle").exists()  # replaced by the fresh JSON
    saved = json.loads((root / "data" / "snapshot.json").read_text())
    assert saved["fetched_utc"] == m["fetched_utc"] and set(saved["sources"]) == {"active", "fengyun-1c", "iridium-33", "cosmos-2251"}


def test_refresh_is_all_or_nothing(sandbox):
    mod, root = sandbox
    fetch, _ = fake_fetch_factory()

    def flaky(url):
        if "cosmos" in url:
            return "<html>error page</html>"
        return fetch(url)

    with pytest.raises(RuntimeError):
        mod.refresh(fetch=flaky, now=dt.datetime(2026, 10, 28, tzinfo=dt.timezone.utc))
    assert not (root / "active.json").exists() and not (root / "data" / "snapshot.json").exists()
    assert (root / "data" / "debris" / "fengyun-1c.tle").read_text() == "old stale file"
    short, _ = fake_fetch_factory(n_active=10)
    with pytest.raises(RuntimeError, match="only 10"):
        mod.refresh(fetch=short, now=dt.datetime(2026, 10, 28, tzinfo=dt.timezone.utc))


def test_refresh_respects_the_two_hour_rule_and_dry_run(sandbox):
    mod, root = sandbox
    fetch, calls = fake_fetch_factory()
    t0 = dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc)
    mod.refresh(fetch=fetch, now=t0)
    with pytest.raises(RuntimeError, match="only 0.5 h old"):
        mod.refresh(fetch=fetch, now=t0 + dt.timedelta(minutes=30))
    n_before = len(calls)
    mod.refresh(fetch=fetch, now=t0 + dt.timedelta(minutes=30), force=True, dry_run=True)
    assert len(calls) == n_before + 4
    mod.refresh(fetch=fetch, now=t0 + dt.timedelta(hours=3))  # allowed again


# ---- consistency check
def frame(groups):
    rows = []
    for name, epoch, n in groups:
        for i in range(n):
            rows.append({"GROUP": name, "EPOCH_DT": pd.Timestamp(epoch) - pd.Timedelta(hours=i)})
    return pd.DataFrame(rows)


def test_health_flags_a_data_set_that_is_out_of_step():
    ok = snapshot.health(frame([(None, "2026-10-27", 50), ("Iridium-33", "2026-10-27", 20)]))
    assert ok["issues"] == []
    bad = snapshot.health(frame([(None, "2026-10-27", 50), ("Cosmos-2251", "2026-10-10", 20)]))
    assert any("Cosmos-2251" in i and "17.0 days" in i for i in bad["issues"])
    assert set(bad["table"]["set"]) == {"Active catalogue", "Cosmos-2251"}


def test_health_flags_stale_share():
    df = frame([(None, "2026-10-27", 40)])
    df.loc[:9, "EPOCH_DT"] = pd.Timestamp("2026-10-01")  # a quarter of the objects are weeks old
    assert any("more than 7 days older" in i for i in snapshot.health(df)["issues"])


def test_real_data_is_consistent_enough():
    df = orbits.catalog_from_records(debris.merge_records(orbits.load_records(), debris.load_debris_records()))
    df["GROUP"] = df["GROUP"] if "GROUP" in df else None
    assert snapshot.health(df)["table"]["objects"].sum() == len(df)


def test_health_reports_age_against_now():
    h = snapshot.health(frame([(None, "2026-10-07", 10)]), now=pd.Timestamp("2026-10-28"))
    assert h["newest_age_days"] == pytest.approx(21.0, abs=0.1)
