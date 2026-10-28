"""Re-Entry & Debris Mitigation Assistant (Streamlit UI). Physics lives in the other modules."""
import base64
import html
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sgp4.api import jday

import charts
import conjunction
import dataset
import decay
import footprint
import globe
import history
import llm
import orbits
import triage

ASSETS = Path(__file__).parent / "assets"


def _data_uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


st.set_page_config(page_title="Orbisk: Re-Entry & Debris Mitigation", page_icon=str(ASSETS / "orbisk-icon.png"), layout="wide")

WATCHLIST_SIZE = 50  # the watchlist always shows the 50 highest-risk active satellites


@st.cache_data
def get_data():
    """Active satellites + every debris group, with the stored closest-approach results attached."""
    cat, recs = dataset.load_catalogue()
    ca = dataset.load_closest_approaches()
    meta = {k: ca[k] for k in ("start_utc", "window_hours", "threshold_km", "top")} if ca else None
    return dataset.attach_closest_approaches(cat, ca), recs, meta


@st.cache_data
def get_history():
    return history.load_history()


df, records, ca_meta = get_data()
hist_all = get_history()
components_df = decay.risk_components(df)

_LOGO_CSS = """<style>
.orbisk-header { display: flex; align-items: center; gap: 1rem; margin: -1rem 0 .75rem; }
.orbisk-header img { height: 52px; width: auto; }
.orbisk-header .for-light { display: none; }
.orbisk-header .tagline { font-size: 1.05rem; line-height: 1.25; opacity: .75; border-left: 1px solid rgba(128,128,128,.45); padding-left: 1rem; }
@media (max-width: 640px) { .orbisk-header .tagline { display: none; } }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) .orbisk-header .for-dark { display: none; }
  :root:not([data-theme="dark"]) .orbisk-header .for-light { display: inline; }
}
</style>"""
# one unindented line of HTML: indented or multi-line HTML is rendered by markdown as a code block
st.markdown(
    _LOGO_CSS
    + '<div class="orbisk-header">'
    + f'<img class="for-dark" alt="Orbisk" src="{_data_uri("orbisk-logo-for-dark.png")}">'
    + f'<img class="for-light" alt="Orbisk" src="{_data_uri("orbisk-logo-for-light.png")}">'
    + '<div class="tagline">Re-entry &amp; debris<br>mitigation assistant</div></div>',
    unsafe_allow_html=True,
)

tab_watch, tab_track, tab_triage = st.tabs(["Watchlist", "Re-entry Tracker", "Conjunctions & Triage"])


@st.cache_data(ttl=300)
def watchlist_paths(norad_ids: tuple, tiers: tuple = (), risks: tuple = ()):
    out = []
    for k, nid in enumerate(norad_ids):
        rec = records[nid]
        try:
            p = orbits.orbit_path_xyz(rec, n_points=120)
        except Exception:
            continue
        row = df[df["NORAD_CAT_ID"] == nid].iloc[0]
        p["name"] = rec["OBJECT_NAME"]
        p["hover"] = f"{rec['OBJECT_NAME']}<br>perigee {row['perigee_km']:.0f} km · apogee {row['apogee_km']:.0f} km"
        if tiers:
            p["tier"] = int(tiers[k])
            p["hover"] += f"<br>risk {risks[k]:.0f}"
        out.append(p)
    return out


@st.cache_data(ttl=300)
def debris_orbit_paths(debris_ids: tuple):
    """Orbit lines for the given debris fragments (empty tuple = none)."""
    return orbits.orbit_paths_fast([records[int(i)] for i in debris_ids], n_points=48) if debris_ids else []


def label(norad_id: int) -> str:
    r = records[norad_id]
    return f"{r['OBJECT_NAME']} ({norad_id})"


def _secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file
        return None


def ai_status():
    """(ready, client_key, message). client_key is the session-pasted key, or None to use the configured one."""
    pasted = st.session_state.get("pasted_key") or ""
    configured = _secret("ANTHROPIC_API_KEY")
    if not (pasted or configured):
        return False, None, "AI is off: add an API key in the assistant (top right) to enable this."
    if configured and not pasted:
        gate = _secret("ASSISTANT_PASSWORD")
        if gate and st.session_state.get("ai_gate") != gate:
            return False, None, "Enter the access code in the assistant (top right) to enable AI features."
    return True, pasted or None, None


def ai_guarded(fn):
    """Run one model call with the shared per-session request cap and friendly errors. None on failure."""
    ready, key, msg = ai_status()
    if not ready:
        st.info(msg)
        return None
    max_req = int(_secret("ASSISTANT_MAX_REQUESTS") or 30)
    if st.session_state.setdefault("ai_requests", 0) >= max_req:
        st.error("Session request limit reached. Reload the page to reset it.")
        return None
    st.session_state["ai_requests"] += 1
    try:
        return fn(llm.make_client(key))
    except Exception as exc:
        st.error(llm.friendly_error(exc))
        return None


def start_time(mode: str):
    """(julian_date, timestamp) for 'Snapshot time' (newest element epoch) or 'Now'."""
    ts = pd.Timestamp.now("UTC").tz_localize(None) if mode == "Now" else df["EPOCH_DT"].max()
    jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
    return jd + fr, ts


# ---------------------------------------------------------------- Watchlist
with tab_watch:
    st.subheader("Watchlist")
    c_mode, c_n = st.columns([3, 1])
    mode = c_mode.selectbox("Rank by", ["Risk score: highest first", "Risk score: lowest first",
                                      "Soonest estimated re-entry"],
                            help="The watchlist draws on the 50 highest-risk active satellites; this orders them and the first "
                                 "'Objects' of that order are shown.")
    n_show = c_n.number_input("Objects", 10, WATCHLIST_SIZE, WATCHLIST_SIZE, step=10,
                              help="How many to show: the first N of the 50 highest-risk satellites in the chosen order.")
    n_deb_total = int((df["source"] == "debris").sum())
    show_debris = st.toggle("Show debris orbits on globe", value=False,
                            help="Debris is not part of the watchlist; this only draws debris orbits on the globe for context.")
    n_deb_show = 0
    if show_debris:
        n_deb_show = st.slider("Debris orbits to draw", 50, n_deb_total, min(50, n_deb_total), step=50,
                               help="Highest-risk debris first.")

    weights = decay.DEFAULT_WEIGHTS  # fixed: not adjustable in the app

    scored = df.copy()
    scored["risk"] = decay.risk_score(components_df, weights)

    def sort_by_mode(frame: pd.DataFrame) -> pd.DataFrame:
        if mode.startswith("Risk"):
            return frame.sort_values("risk", ascending=mode.endswith("lowest first"))
        return frame.sort_values("decay_days", na_position="last")  # soonest estimated re-entry

    sats = scored[(scored["source"] == "active") & scored["leo"]]  # active LEO satellites only (not debris, not GEO/MEO)
    # the pool is always the WATCHLIST_SIZE highest-risk satellites; the order option sorts them and Objects shows the first N
    top = sort_by_mode(sats.sort_values("risk", ascending=False).head(WATCHLIST_SIZE)).head(int(n_show))

    levels = decay.traffic_light(sats["risk"]).loc[top.index]  # level = risk rank among the satellites, whatever is listed
    LEVEL_LABEL = {"red": "🔴 High", "yellow": "🟡 Medium", "green": "🟢 Low"}
    hours_txt = ["—" if pd.isna(h) else f"{h:.1f} h" for h in top["ca_hours"]]
    table = pd.DataFrame({
        "Level": levels.map(LEVEL_LABEL),
        "Name": top["OBJECT_NAME"],
        "Perigee (km)": top["perigee_km"], "Apogee (km)": top["apogee_km"], "Inclination (°)": top["INCLINATION"],
        "Est. time to re-entry": top["decay_days"].map(decay.format_days),
        "Range (±×2)": [decay.format_range(lo, hi) for lo, hi in zip(top["decay_days_low"], top["decay_days_high"])],
        "Closest approach (km)": top["ca_miss_km"], "Rel. speed (km/s)": top["ca_speed_km_s"],
        "Time to closest approach": hours_txt,
        "Risk score": top["risk"],
    })
    table_ids = tuple(int(i) for i in top["NORAD_CAT_ID"])
    debris_ranked = scored[scored["source"] == "debris"].sort_values("risk", ascending=False)
    deb_ids = tuple(int(i) for i in debris_ranked["NORAD_CAT_ID"])[:int(n_deb_show)]  # debris only appears as globe orbits
    tier_idx = tuple({"red": 0, "yellow": 1, "green": 2}[l] for l in levels)
    components.html(globe.orbit_html(watchlist_paths(table_ids, tier_idx, tuple(float(r) for r in top["risk"])),
                                     debris=debris_orbit_paths(deb_ids)), height=660)
    bar_hex = {"red": "#e5484d", "yellow": "#e0a800", "green": "#30a46c"}

    def risk_bar(score: float, level: str) -> str:
        """Progress-bar image (data URI): track + fill in the level colour + the score. The leading comment holds the
        zero-padded score so sorting the column orders rows by score."""
        fill = 150 * score / 100
        svg = (f'<!--{round(score * 10):04d}--><svg xmlns="http://www.w3.org/2000/svg" width="210" height="28" viewBox="0 0 210 28">'
               f'<rect x="0" y="11" width="150" height="6" rx="3" fill="#808080" fill-opacity="0.28"/>'
               f'<rect x="0" y="11" width="{fill:.1f}" height="6" rx="3" fill="{bar_hex[level]}"/>'
               f'<text x="162" y="18" font-family="Source Sans Pro, Arial, sans-serif" font-size="12" font-weight="600" '
               f'fill="{bar_hex[level]}">{score:.0f}</text></svg>')
        return "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="")

    shown_table = table.assign(**{"Risk score": [risk_bar(s, l) for s, l in zip(top["risk"], levels)]})
    event = st.dataframe(
        shown_table, width="stretch", hide_index=True, key=f"watch_table_{st.session_state.get('_watch_gen', 0)}", on_select="rerun",
        selection_mode=["single-row", "single-cell"],  # click any cell (or tick the row) to open the object
        column_config={
            "Perigee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Apogee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Inclination (°)": st.column_config.NumberColumn(format="%.2f"),
            "Closest approach (km)": st.column_config.NumberColumn(format="%.2f"),
            "Rel. speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
            "Risk score": st.column_config.ImageColumn("Risk score", width="medium"),
        },
    )
    st.download_button("Download CSV", data=table.assign(**{"NORAD ID": top["NORAD_CAT_ID"].astype(int)})
                       .assign(Level=levels.map({"red": "High", "yellow": "Medium", "green": "Low"}))
                       .to_csv(index=False).encode("utf-8"),
                       file_name="orbisk_watchlist.csv", mime="text/csv")
    # selecting a row opens that object in the Re-entry Tracker (set once per new selection so the picker stays usable)
    picked = list(event.selection.rows) or [c[0] if isinstance(c, (tuple, list)) else c.get("row") for c in event.selection.cells]
    picked_id = table_ids[picked[0]] if picked and picked[0] < len(table_ids) else None
    switch_target = "Re-entry Tracker" if picked_id is not None and picked_id != st.session_state.get("_watch_picked") else None
    st.session_state["_watch_picked"] = picked_id
    if switch_target:
        st.session_state["tracker_obj"] = picked_id
        st.session_state["_watch_gen"] = st.session_state.get("_watch_gen", 0) + 1  # fresh table next run: the same row can be clicked again
    if ca_meta is None or not top["ca_screened"].all():
        st.caption("Close-approach columns are filled for the highest-risk objects only (`tools/compute_conjunctions.py`); "
                   "— means not screened or no approach within the threshold.")

# ---------------------------------------------------------------- Tracker
with tab_track:
    # the same 50 highest-risk active satellites as the watchlist pool, highest risk first
    ids = sats.sort_values("risk", ascending=False).head(WATCHLIST_SIZE)["NORAD_CAT_ID"].astype(int).tolist()
    default = 0
    norad = st.selectbox("Object", ids, format_func=label, key="tracker_obj")
    row = sats[sats["NORAD_CAT_ID"] == norad].iloc[0]
    record = records[int(norad)]
    name = record["OBJECT_NAME"]
    level = decay.traffic_light(sats["risk"]).loc[row.name]
    comp_row = components_df.loc[row.name]

    t_orbit, t_decay, t_foot, t_risk_bd = st.tabs(["Orbit", "Decay", "Re-entry footprint", "Risk breakdown"])

    try:
        pos = orbits.current_position(record)
        track = orbits.ground_track(record, n_orbits=3)
        propagate_error = None
    except Exception as exc:  # e.g. decayed / bad elements
        pos = track = None
        propagate_error = exc

    with t_orbit:
        if propagate_error is not None:
            st.error(f"Could not propagate {name}: {propagate_error}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latitude", f"{pos['lat']:.2f}°")
            c2.metric("Longitude", f"{pos['lon']:.2f}°")
            c3.metric("Altitude", f"{pos['alt_km']:.0f} km")
            c4.metric("Perigee / Apogee", f"{row['perigee_km']:.0f} / {row['apogee_km']:.0f} km")
            path = orbits.orbit_path_xyz(record, n_points=360)
            path["name"] = name
            components.html(globe.orbit_html([path], height=600, highlight=True, autoplay=False), height=610)  # paused at "now": matches the ground track

        # ---- summary card (below the globe)
        with st.container(border=True):
            h1, h2 = st.columns([4, 1])
            h1.markdown(f"### {name}")
            h1.caption(f"NORAD ID {int(norad)} · {row['type'].capitalize()}")
            h2.badge({"red": "High risk", "yellow": "Medium risk", "green": "Low risk"}[level],
                     color={"red": "red", "yellow": "orange", "green": "green"}[level])
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Risk score", f"{row['risk']:.0f} / 100")
            s2.metric("Est. time to re-entry", decay.format_days(row["decay_days"]))
            s3.metric("Rough range", decay.format_range(row["decay_days_low"], row["decay_days_high"]))
            if bool(row.get("ca_screened", False)) and pd.notna(row.get("ca_miss_km")):
                s4.metric("Closest approach", f"{row['ca_miss_km']:.2f} km",
                          help=f"With {row['ca_with']}, {row['ca_hours']:.1f} h after the snapshot, at {row['ca_speed_km_s']:.1f} km/s.")
            else:
                s4.metric("Closest approach", "—", help="Not screened, or no approach inside the threshold.")
            if st.button("Screen this object", key="screen_this", type="primary",
                         help="Opens Conjunctions & Triage with this object selected."):
                st.session_state["tri_obj"] = int(norad)
                switch_target = "Conjunctions & Triage"

    with t_decay:
        h0 = float(decay.mean_altitude_km(row["MEAN_MOTION"]))
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Est. time to re-entry", decay.format_days(row["decay_days"]))
        d2.metric("Rough range", decay.format_range(row["decay_days_low"], row["decay_days_high"]))
        d3.metric("Current decay rate", f"{row['decay_km_day']:.3f} km/day" if np.isfinite(row["decay_km_day"]) else "—")
        h_obj = history.object_history(hist_all, int(norad))
        obs = history.observed_decay_rate(h_obj)
        d4.metric("Observed perigee trend", f"{obs:+.3f} km/day" if obs is not None else "n/a",
                  help="Needs at least two recorded snapshots spanning a day or more.")
        st.caption(f"Status: **{row['decay_status']}**. This is a rough estimate (typically good to a factor of 2 or worse), "
                   "not a re-entry prediction; it cannot anticipate solar activity changes or manoeuvres.")
        st.plotly_chart(charts.decay_figure(h0, float(row["decay_km_day"]), h_obj), width="stretch")

    with t_foot:
        f1, f2 = st.columns(2)
        f1.metric("Latitudes overflown", f"±{footprint.max_latitude(row['INCLINATION']):.1f}°")
        f2.metric("Share of Earth's surface", f"{footprint.surface_fraction(row['INCLINATION']) * 100:.0f}%")
        st.plotly_chart(charts.footprint_figure(float(row["INCLINATION"])), width="stretch")
        if propagate_error is None:
            st.markdown("**Ground track** (next 3 orbits)")
            st.plotly_chart(globe.ground_track_figure(track, pos), width="stretch")
        st.caption("An object can only come down within the latitude band set by its inclination (red), and spends most of its time "
                   "near the band's edges (right). Where it actually re-enters depends on the final orbits and cannot be "
                   "predicted days ahead; this shows where it *can* pass, not where it *will* land.")

    with t_risk_bd:
        st.metric("Risk score", f"{row['risk']:.1f} / 100")
        st.plotly_chart(charts.risk_breakdown_figure(comp_row, weights, level), width="stretch")
        st.caption("Each bar is one weighted term of the score: the grey bar is the most it could add (its weight), the colour bar is "
                   "what this object gets. The seven terms add up to the risk score.")

# ---------------------------------------------------------------- Triage desk
fmt_utc = lambda jd: conjunction.jd_to_datetime(jd).strftime("%Y-%m-%d %H:%M:%S")


def events_table(ev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Object": ev["name"], "NORAD ID": ev["norad_id"],
        "Closest approach (UTC)": [fmt_utc(j) for j in ev["tca_jd"]],
        "Miss distance (km)": ev["miss_km"], "Relative speed (km/s)": ev["rel_speed_km_s"],
        "Its perigee (km)": ev["other_perigee_km"], "Its apogee (km)": ev["other_apogee_km"],
        **({"Debris set": ev["other_group"].fillna("")} if "other_group" in ev else {}),
    })


@st.cache_data(ttl=300, show_spinner=False)
def conjunction_view_html(asset: int, partners: tuple, tcas: tuple, misses: tuple, start: str) -> str:
    """Globe with the screened satellite's orbit (cyan), each close partner's orbit and a marker where the pass happens.
    Orbit and marker colours follow the miss-distance bands of the cards."""
    from sgp4.api import jday
    t0 = orbits.TS.utc(*pd.Timestamp(start).timetuple()[:6])
    theta = np.radians(t0.gmst * 15.0) % (2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    tier_of = lambda miss: 0 if miss < 5 else (1 if miss < 15 else 2)
    paths = []
    for nid, tier in [(asset, 3)] + [(p, tier_of(m)) for p, m in zip(partners, misses)]:
        try:
            p = orbits.orbit_path_xyz(records[int(nid)], n_points=180, start=t0)
        except Exception:
            continue
        p["name"], p["tier"] = records[int(nid)]["OBJECT_NAME"], tier
        p["hover"] = records[int(nid)]["OBJECT_NAME"]
        paths.append(p)
    sat = conjunction.to_satrec(records[int(asset)])
    events = []
    for nid, tca, miss in zip(partners, tcas, misses):
        whole = np.floor(tca - 0.5) + 0.5  # sgp4 wants the Julian date split into day and fraction
        e, r, _ = sat.sgp4(float(whole), float(tca - whole))
        if e != 0:
            continue
        events.append({"x": r[0] * c + r[1] * s, "y": -r[0] * s + r[1] * c, "z": r[2], "tier": tier_of(miss),
                       "hover": f"{records[int(nid)]['OBJECT_NAME']}<br>{miss:.2f} km at {fmt_utc(tca)} UTC"})
    return globe.orbit_html(paths, height=560, events=events)


def miss_band(miss_km: float) -> tuple[str, str]:
    """(label, colour) for a miss distance: red under 5 km, yellow under 15 km, green otherwise."""
    if miss_km < 5:
        return "Under 5 km", "#e5484d"
    if miss_km < 15:
        return "5 to 15 km", "#e0a800"
    return "15 km or more", "#30a46c"


EVENT_CONFIG = {
    "NORAD ID": st.column_config.NumberColumn(format="%d"),
    "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
    "Relative speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
    "Its perigee (km)": st.column_config.NumberColumn(format="%.0f"),
    "Its apogee (km)": st.column_config.NumberColumn(format="%.0f"),
}


with tab_triage:
    st.subheader("Conjunctions & triage")
    st.markdown("Pick one of the watchlist satellites, a time window and a distance. Orbisk flies that satellite and every other "
                "satellite and debris fragment in a similar altitude band forward in time and lists the passes closer than your "
                "distance. The closest ones appear first as colour-coded cards. This is a screening aid, not a collision probability.")
    asset_ids = ids  # the 50 highest-risk active satellites, same as the watchlist
    with st.form("tri_asset_form"):
        a1, a2, a3 = st.columns([3, 1, 1])
        asset = a1.selectbox("Object to screen", asset_ids, format_func=label, key="tri_obj")
        days = a2.slider("Window (days)", 1, 30, 2)
        hours = days * 24
        thr = a3.slider("Report within (km)", 1, 100, 25)
        run = st.form_submit_button("Run screening")
    if run:
        jd0, ts0 = start_time("Snapshot time")
        with st.spinner("Propagating candidates…"):
            try:
                ev, n = triage.screen_asset(records[int(asset)], df, records, jd0, hours=hours, threshold_km=thr,
                                            against="all")
            except Exception as exc:
                st.error(f"Screening failed: {exc}")
            else:
                st.session_state["tri_asset"] = {"asset": int(asset), "events": ev, "n": n, "hours": ev.attrs.get("window_hours_effective", hours), "asked": hours, "thr": thr,
                                                 "start": ts0.strftime("%Y-%m-%d %H:%M")}
    res = st.session_state.get("tri_asset")
    if res:
        ev = res["events"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects in a similar altitude band", f"{res['n']:,}")
        m2.metric(f"Passes within {res['thr']} km", f"{len(ev)}")
        m3.metric("Closest pass", f"{ev['miss_km'].min():.1f} km" if len(ev) else "none")
        if res["hours"] < res["asked"] - 1:  # only shown when the screened object's orbit ends inside the window
            st.caption(f"This object's orbit ends after {res['hours'] / 24:.1f} days, so only that part was screened.")
        if not len(ev):
            st.success("No close approaches against any catalogue object inside the threshold in this window.")
        else:
            table_ev = events_table(ev)
            closest = ev.sort_values("miss_km").head(6)
            st.markdown("**Closest passes**")
            card_cols = st.columns(3)
            for k, r in enumerate(closest.itertuples()):
                band = miss_band(r.miss_km)
                with card_cols[k % 3]:
                    st.markdown(
                        f'<div style="border:1px solid rgba(128,128,128,.35);border-left:6px solid {band[1]};border-radius:8px;'
                        f'padding:.6rem .8rem;margin-bottom:.6rem">'
                        f'<div style="font-weight:600">{html.escape(str(r.name))}</div>'
                        f'<div style="font-size:.8rem;opacity:.7">NORAD {int(r.norad_id)}</div>'
                        f'<div style="font-size:1.6rem;font-weight:600;color:{band[1]}">{r.miss_km:.2f} km</div>'
                        f'<div style="font-size:.8rem">{band[0]} · {fmt_utc(r.tca_jd)} UTC · {r.rel_speed_km_s:.1f} km/s</div></div>',
                        unsafe_allow_html=True)
            if len(ev) > len(closest):
                st.caption(f"Showing the {len(closest)} closest of {len(ev)} passes; all of them are in the table below.")
            st.markdown("**Orbit view**")
            components.html(conjunction_view_html(int(res["asset"]), tuple(int(n) for n in closest["norad_id"]),
                                                  tuple(float(j) for j in closest["tca_jd"]),
                                                  tuple(float(m) for m in closest["miss_km"]), res["start"]), height=570)
            st.caption("Cyan: the screened satellite. ◆ marks each pass, coloured like the cards.")
            st.dataframe(table_ev, width="stretch", hide_index=True, column_config=EVENT_CONFIG)
            st.download_button("Download CSV", data=table_ev.to_csv(index=False).encode("utf-8"),
                               file_name=f"orbisk_conjunctions_{res['asset']}.csv", mime="text/csv", key="tri_csv")

# floating chat launcher (top-right, below Streamlit's toolbar); the popover opens downward and closes with the same button
st.markdown(
    """<style>
    .st-key-ai_fab { position: fixed; top: 3.75rem; right: 1.5rem; z-index: 1000; width: auto !important; }
    .st-key-ai_fab button { width: 3.5rem; height: 3.5rem; padding: 0; border-radius: 16px; border: 0; font-size: 0;
        background: url(__ICON__) center / cover no-repeat; box-shadow: 0 2px 10px rgba(0,0,0,.35); }
    .st-key-ai_fab button > * { display: none; }
    .st-key-ai_fab { touch-action: none; cursor: grab; }
    [data-testid="stPopoverBody"] { width: min(520px, 92vw); max-height: calc(100vh - 8rem); overflow-y: auto; }
    </style>""".replace("__ICON__", _data_uri("orbisk-icon.png")),
    unsafe_allow_html=True,
)
with st.container(key="ai_fab"):
    with st.popover("Ask the Orbisk assistant"):
        st.caption("Optional. The app is fully offline without this. When you add an Anthropic API key the assistant can "
                   "explain results in plain language. It never calculates orbits: it only sees numbers this app computed, "
                   "or looks data up through the app's own functions.")
        pasted = st.text_input("Anthropic API key", type="password", key="pasted_key",
                               help="Kept only in this browser session; not stored or logged.")
        configured = _secret("ANTHROPIC_API_KEY")
        api_key = pasted or configured
        key_from_config = bool(configured) and not pasted

        if not api_key:
            st.info("AI features are off. Paste a key above, or set `ANTHROPIC_API_KEY` as an environment variable or in "
                    "Streamlit secrets. Everything else in the app works without it.")
        else:
            allowed = True
            if key_from_config:
                gate = _secret("ASSISTANT_PASSWORD")
                if gate:
                    allowed = st.text_input("Access code", type="password", key="ai_gate") == gate
                    if not allowed:
                        st.warning("Enter the access code to use the assistant.")
                else:
                    st.warning("This deployment uses a key from its configuration, so anyone with the link can spend it. "
                               "Set `ASSISTANT_PASSWORD` to require an access code.")
            if allowed:
                max_req = int(_secret("ASSISTANT_MAX_REQUESTS") or 30)
                used = st.session_state.setdefault("ai_requests", 0)
                model = st.selectbox("Model", list(llm.MODELS), format_func=llm.MODELS.get, key="ai_model")
                st.caption(f"{used}/{max_req} requests used in this session.")
                ctx = llm.DataContext(df, records, components_df, weights, hist_all)

                guarded = ai_guarded

                t_chat, t_brief, t_risk, t_rec = st.tabs(["Ask the data", "Object briefing", "Risk explanation", "Recommended action"])
                for tab, kind, task, button in ((t_brief, "brief", llm.BRIEFING_TASK, "Generate briefing"),
                                                (t_risk, "risk", llm.RISK_TASK, "Explain risk score"),
                                                (t_rec, "rec", llm.RECOMMEND_TASK, "Recommend next steps")):
                    with tab:
                        pick = st.selectbox("Object", ids, index=default, format_func=label, key=f"ai_pick_{kind}")
                        facts = llm.object_facts(ctx, int(pick))
                        if st.button(button, key=f"ai_go_{kind}"):
                            with st.spinner("Asking the model…"):
                                res = guarded(lambda c: llm.explain(c, model, facts, task))
                            if res:
                                st.session_state[f"ai_out_{kind}"] = (pick, res)
                        saved = st.session_state.get(f"ai_out_{kind}")
                        if saved and saved[0] == pick:
                            st.markdown(saved[1][0])
                            st.caption(f"{saved[1][1]['input_tokens']} input / {saved[1][1]['output_tokens']} output tokens")
                        with st.expander("Facts sent to the model (computed by this app)"):
                            st.json(facts)

                with t_chat:
                    st.caption("Examples: *Which red satellites have the closest approaches?* · *Why is STARLINK-1433 ranked so high?* · "
                               "*What should we do about the ISS?* · *How is the risk score calculated?*")
                    chat_hist = st.session_state.setdefault("ai_chat", [])
                    for m in chat_hist:
                        with st.chat_message(m["role"]):
                            st.markdown(m["content"])
                            if m.get("trace"):
                                st.caption("Looked up: " + ", ".join(f"{t['tool']}({t['input']})" for t in m["trace"]))
                    top_sat = sats.sort_values("risk", ascending=False).iloc[0]
                    suggestions = {  # button label -> question sent to the assistant
                        "Why is this object ranked first?": f"Why is {top_sat['OBJECT_NAME']} (NORAD {int(top_sat['NORAD_CAT_ID'])}) "
                                                           "ranked first by risk score?",
                        "Which satellites re-enter soonest?": "Which satellites have the soonest estimated re-entry?",
                        "What is perigee and apogee?": "What are perigee and apogee, and why do they matter for re-entry?",
                    }
                    asked = None
                    holder = st.empty()
                    if not chat_hist:
                        with holder.container():
                            for i, (label_, question) in enumerate(suggestions.items()):
                                if st.button(label_, key=f"ai_suggest_{i}", width="stretch"):
                                    asked = question
                        if asked:
                            holder.empty()  # the chat has started: drop the suggestions
                    q = st.chat_input("Ask about the catalogue…") or asked
                    if q:
                        with st.chat_message("user"):
                            st.markdown(q)
                        with st.chat_message("assistant"):
                            with st.spinner("Working…"):
                                plain = [{"role": m["role"], "content": m["content"]} for m in chat_hist]
                                res = guarded(lambda c: llm.chat(c, model, ctx, plain, q))
                            if res:
                                answer, trace, usage = res
                                st.markdown(answer)
                                if trace:
                                    st.caption("Looked up: " + ", ".join(f"{t['tool']}({t['input']})" for t in trace))
                                st.caption(f"{usage['input_tokens']} input / {usage['output_tokens']} output tokens")
                                chat_hist += [{"role": "user", "content": q},
                                              {"role": "assistant", "content": answer, "trace": trace}]
                    if chat_hist and st.button("Clear chat"):
                        st.session_state["ai_chat"] = []
                        st.rerun()

with st.expander("How these numbers are computed"):
    st.markdown(
        "- **Perigee / apogee:** a from mean motion via Kepler's third law (n²a³ = μ); altitude = a(1∓e) − 6378.137 km.\n"
        "- **Positions and orbits:** SGP4 (via skyfield) on the stored mean elements; the globe is drawn in Earth-fixed axes with the Earth rotating at its real rate.\n"
        "- **Decay estimate:** the measured `MEAN_MOTION_DOT` (×2, since the field is ṅ/2) gives da/dt = −(2/3)·a·ṅ/n. "
        "That rate is scaled by air density (exponential atmosphere table) and integrated down to 120 km. "
        "Requires positive `BSTAR` (drag term). Uncertainty is at least a factor of 2.\n"
        "- **Risk score (0-100):** fixed weighted blend of seven terms scaled 0-1: perigee 20% (150 km = 1, 700 km or more = 0), "
        "decay time 20% (10 days or less = 1, 10 years or more = 0), inclination 10% (share of Earth's surface overflown), "
        "object type 5% (debris 1.0, rocket body 0.8, payload 0.3), miss distance 25% (closest approach to any satellite or debris in 24 h "
        "from the snapshot time: 1 km = 1, 25 km or more = 0), relative speed 10% (km/s over 15) and time to closest approach 10% "
        "(within 6 h = 1, 24 h or more = 0); the last two only count when the miss is under 25 km. Close-approach results are precomputed "
        "for the highest-risk objects.\n"
        "- **Colour levels:** by risk rank among the active satellites: the 10 highest-risk are red, the next 30 yellow, all others green "
        "(the watchlist draws on the 50 highest-risk satellites, so all 50 show 10 / 30 / 10; fewer are the first N of them in the chosen order). Debris is not on the watchlist; it can be drawn on the globe.\n"
        "- **Footprint:** latitude band ±min(i, 180−i); time per latitude from sin φ = sin i · sin u.\n"
        "- **Conjunctions:** coarse-grid SGP4 screening with 0.5 s refinement of local minima, against every satellite and debris fragment.\n"
        "- **Scope:** low Earth orbit (LEO) only: the watchlist, tracker, screening and assistant cover active satellites with a mean altitude and perigee below 2,000 km. "
        "Objects in higher orbits are not listed (a higher orbit that crosses LEO still counts as a close-approach partner).\n"
        "- **Debris:** the debris groups (TLE or CelesTrak JSON files in data/debris) are parsed into the same record format. They are not on the watchlist "
        "or in the pickers; they are only drawn as orbits on the Watchlist globe and used as close-approach partners when an active satellite is screened "
        "(Conjunctions & Triage, and the close-approach terms of the risk score). Altitude and risk figures are for active satellites only."
        "Altitude bands are the densest ranges of the active catalogue, found from the data.\n"
        "- **Assistant (optional):** off unless an API key is supplied. The model only receives numbers computed above, or calls the app's own lookup functions; it never computes orbits.\n"
        "- **Limits:** one element set per object (no history until snapshots are added), mean elements are only km-accurate, "
        "and active satellites that manoeuvre invalidate drag-based decay estimates."
    )

# make the launcher draggable: script runs in a same-origin iframe and drives the parent page; the position is kept in
# a <style> tag in the parent head (survives Streamlit re-renders) and in localStorage
components.html("""<script>
const P = window.parent, D = P.document, KEY = "orbisk_fab_pos";
if (!P.__orbiskDrag) {
  P.__orbiskDrag = true;
  const style = D.createElement("style");
  D.head.appendChild(style);
  const apply = (x, y) => {
    const w = 56, h = 56;
    x = Math.min(Math.max(0, x), P.innerWidth - w); y = Math.min(Math.max(0, y), P.innerHeight - h);
    style.textContent = `.st-key-ai_fab{left:${x}px !important;top:${y}px !important;right:auto !important;}`;
    return [x, y];
  };
  try { const s = JSON.parse(P.localStorage.getItem(KEY)); if (s) apply(s[0], s[1]); } catch (e) {}
  let drag = null;
  D.addEventListener("pointerdown", (e) => {
    const fab = e.target.closest(".st-key-ai_fab");
    if (!fab || e.button > 0) return;
    const r = fab.getBoundingClientRect();
    drag = {dx: e.clientX - r.left, dy: e.clientY - r.top, x0: e.clientX, y0: e.clientY, moved: false};
  }, true);
  D.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 6) return;
    drag.moved = true;
    drag.pos = apply(e.clientX - drag.dx, e.clientY - drag.dy);
    e.preventDefault();
  }, true);
  D.addEventListener("pointerup", () => {
    if (drag && drag.moved) {
      try { P.localStorage.setItem(KEY, JSON.stringify(drag.pos)); } catch (e) {}
      const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };  // a drag must not open the popup
      D.addEventListener("click", swallow, {capture: true, once: true});
      setTimeout(() => D.removeEventListener("click", swallow, true), 300);
    }
    drag = null;
  }, true);
  // turning the "Show debris orbits" toggle off hides the debris in the globe straight away; the new page then
  // arrives after the (slow) rerun and keeps it hidden
  D.addEventListener("click", (e) => {
    const box = e.target.closest('[data-testid="stCheckbox"], [data-testid="stToggle"]');
    if (!box || !box.textContent.includes("Show debris orbits on globe")) return;
    setTimeout(() => {
      const input = box.querySelector("input");
      if (!input || input.checked) return;
      for (const f of D.querySelectorAll("iframe")) {
        try {
          for (const l of f.contentDocument.querySelectorAll("label")) {
            const cb = l.querySelector('input[type="checkbox"]');
            if (cb && cb.checked && /Debris orbits \\(/.test(l.textContent)) cb.click();
          }
        } catch (err) {}
      }
    }, 50);
  }, true);
  P.addEventListener("resize", () => { try { const s = JSON.parse(P.localStorage.getItem(KEY)); if (s) apply(s[0], s[1]); } catch (e) {} });
}
</script>""", height=0)

if switch_target:  # jump to the chosen tab (Streamlit has no API for this, so click the tab header once the run is done)
    components.html("""<script>
    const D = window.parent.document;
    let tries = 0;
    const go = setInterval(() => {
      tries += 1;
      const busy = D.querySelector('[data-testid="stStatusWidget"]');
      if (!busy || tries > 150) {
        clearInterval(go);
        for (const t of D.querySelectorAll('[role="tab"]')) { if (t.textContent.trim() === "__TARGET__") { t.click(); break; } }
      }
    }, 100);
    </script>""".replace("__TARGET__", switch_target), height=0)
