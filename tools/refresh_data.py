"""Download the active catalogue and every debris group TOGETHER, so all element epochs line up.

Usage:  python tools/refresh_data.py [--force] [--dry-run]

Run it on a machine with internet access, then commit the changed files in data/ and active.json.
It is all-or-nothing: if any download fails or looks malformed, nothing is written. CelesTrak refreshes
its data about every two hours and asks users not to download more often, so the script refuses to run
again within two hours of the last snapshot unless --force is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import debris  # noqa: E402
import history  # noqa: E402
import orbits  # noqa: E402
import snapshot  # noqa: E402

BASE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
MIN_HOURS_BETWEEN_RUNS = 2.0
MIN_ROWS = {"active": 5000}  # sanity floor so an error page or a truncated reply is never saved


def fetch_text(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "reentry-debris-assistant/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def plan() -> dict[str, tuple[str, Path]]:
    """name -> (url, destination file)."""
    out = {"active": (BASE.format(group="active"), ROOT / "active.json")}
    for slug, (_, group) in debris.GROUPS.items():
        out[slug] = (BASE.format(group=group), debris.DEBRIS_DIR / f"{slug}.json")
    return out


def refresh(fetch=fetch_text, force: bool = False, dry_run: bool = False, now: dt.datetime | None = None) -> dict:
    """Download everything first, validate, then write. Returns the manifest that was (or would be) written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    prev = snapshot.load_manifest()
    if prev and not force:
        age_h = (now - dt.datetime.fromisoformat(prev["fetched_utc"])).total_seconds() / 3600.0
        if age_h < MIN_HOURS_BETWEEN_RUNS:
            raise RuntimeError(f"last snapshot is only {age_h:.1f} h old; CelesTrak updates every ~2 h (use --force to override)")

    downloaded: dict[str, tuple[str, list[dict], Path]] = {}
    for name, (url, dest) in plan().items():
        text = fetch(url)
        try:
            recs = debris.parse_omm_json(text)
        except ValueError as exc:
            raise RuntimeError(f"{name}: unexpected reply ({exc}); first 80 chars: {text[:80]!r}") from exc
        if len(recs) < MIN_ROWS.get(name, 1):
            raise RuntimeError(f"{name}: only {len(recs)} usable records, refusing to overwrite the stored data")
        downloaded[name] = (text, recs, dest)

    manifest = {"fetched_utc": now.isoformat(timespec="seconds"),
                "sources": {n: {**snapshot.source_stats(r), "file": str(d.relative_to(ROOT))} for n, (_, r, d) in downloaded.items()}}
    if dry_run:
        return manifest
    for name, (text, _, dest) in downloaded.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        for twin in (".tle", ".txt"):  # an older copy of the same group in TLE form would just be a stale duplicate
            old = dest.with_suffix(twin)
            if old != dest and old.exists() and name != "active":
                old.unlink()
    snapshot.write_manifest(manifest["fetched_utc"], manifest["sources"])
    history.append_snapshot(orbits.load_catalog())
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="ignore the 2-hour minimum between downloads")
    ap.add_argument("--dry-run", action="store_true", help="download and validate but write nothing")
    args = ap.parse_args()
    try:
        m = refresh(force=args.force, dry_run=args.dry_run)
    except Exception as exc:
        sys.exit(f"refresh failed, nothing was written: {exc}")
    for name, s in m["sources"].items():
        print(f"{name:12s} {s['count']:6d} objects, epochs {s['epoch_min'][:16]} .. {s['epoch_max'][:16]}")
    print("dry run only" if args.dry_run else f"snapshot written ({m['fetched_utc']})")
