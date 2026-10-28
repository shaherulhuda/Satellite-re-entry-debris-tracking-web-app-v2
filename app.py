"""Re-Entry & Debris Mitigation Assistant (Streamlit UI). Physics lives in the other modules."""
import os
import urllib.parse

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

st.set_page_config(page_title="Re-Entry & Debris Mitigation Assistant", layout="wide")


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

st.title("Re-Entry & Debris Mitigation Assistant")

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
    c_mode, c_n, c_d = st.columns([3, 1, 1])
    mode = c_mode.radio("Rank by", ["Risk score (highest first)", "Soonest estimated re-entry", "Lowest perigee"],
                        horizontal=True)
    n_show = c_n.number_input("Objects", 10, 200, 50, step=10)
    n_deb_total = int((df["source"] == "debris").sum())
    n_deb_show = c_d.number_input("Debris", 0, n_deb_total, 0, step=50,
                                  help="Debris is not part of the watchlist; this draws that many debris orbits on the globe for context, highest ranked first. 0 = hidden.")

    weights = decay.DEFAULT_WEIGHTS  # fixed: not adjustable in the app

    scored = df.copy()
    scored["risk"] = decay.risk_score(components_df, weights)

    def sort_by_mode(frame: pd.DataFrame) -> pd.DataFrame:
        if mode.startswith("Risk"):
            return frame.sort_values("risk", ascending=False)
        if mode.startswith("Soonest"):
            return frame.sort_values("decay_days", na_position="last")
        return frame.sort_values("perigee_km")

    sats = scored[scored["source"] == "active"]  # the watchlist covers active satellites only, not debris
    ranked = sort_by_mode(sats)
    top = ranked.head(int(n_show))

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
    debris_ranked = sort_by_mode(scored[scored["source"] == "debris"])
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
    st.dataframe(
        shown_table, width="stretch", hide_index=True,
        column_config={
            "Perigee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Apogee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Inclination (°)": st.column_config.NumberColumn(format="%.2f"),
            "Closest approach (km)": st.column_config.NumberColumn(format="%.2f"),
            "Rel. speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
            "Risk score": st.column_config.ImageColumn("Risk score", width="medium"),
        },
    )
    if ca_meta is None or not top["ca_screened"].all():
        st.caption("Close-approach columns are filled for the highest-risk objects only (`tools/compute_conjunctions.py`); "
                   "— means not screened or no approach within the threshold.")

# ---------------------------------------------------------------- Tracker
with tab_track:
    ids = df[df["source"] == "active"].sort_values("OBJECT_NAME")["NORAD_CAT_ID"].astype(int).tolist()  # active satellites only
    default = ids.index(25544) if 25544 in ids else 0
    norad = st.selectbox("Object", ids, index=default, format_func=label)
    row = df[df["NORAD_CAT_ID"] == norad].iloc[0]
    record = records[int(norad)]
    name = record["OBJECT_NAME"]

    try:
        pos = orbits.current_position(record)
        track = orbits.ground_track(record, n_orbits=3)
    except Exception as exc:  # e.g. decayed / bad elements
        st.error(f"Could not propagate {name}: {exc}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latitude", f"{pos['lat']:.2f}°")
        c2.metric("Longitude", f"{pos['lon']:.2f}°")
        c3.metric("Altitude", f"{pos['alt_km']:.0f} km")
        c4.metric("Perigee / Apogee", f"{row['perigee_km']:.0f} / {row['apogee_km']:.0f} km")
        st.caption(f"Position at {pos['time']:%Y-%m-%d %H:%M:%S} UTC (now), propagated from the snapshot elements.")

        st.markdown("**3D orbit** (yellow dot = position now; the animation loops until you press Pause)")
        path = orbits.orbit_path_xyz(record, n_points=360)
        path["name"] = name
        components.html(globe.orbit_html([path], height=600, highlight=True), height=610)

        st.markdown("**Ground track** (next 3 orbits)")
        st.plotly_chart(globe.ground_track_figure(track, pos), width="stretch")

    # ---- decay estimate and history
    st.subheader("Decay estimate")
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
    if len(h_obj) < 2:
        st.caption("Perigee history builds up as snapshots are added: replace `active.json` with fresh data and run "
                   "`python tools/add_snapshot.py`. Only objects with perigee ≤ "
                   f"{history.DEFAULT_MAX_PERIGEE_KM:.0f} km are recorded.")

    # ---- footprint
    st.subheader("Re-entry ground footprint")
    f1, f2 = st.columns(2)
    f1.metric("Latitudes overflown", f"±{footprint.max_latitude(row['INCLINATION']):.1f}°")
    f2.metric("Share of Earth's surface", f"{footprint.surface_fraction(row['INCLINATION']) * 100:.0f}%")
    st.plotly_chart(charts.footprint_figure(float(row["INCLINATION"])), width="stretch")
    st.caption("An object can only come down within the latitude band set by its inclination (red), and spends most of its time "
               "near the band's edges (right). Where it actually re-enters depends on the final orbits and cannot be "
               "predicted days ahead; this shows where it *can* pass, not where it *will* land.")

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


EVENT_CONFIG = {
    "NORAD ID": st.column_config.NumberColumn(format="%d"),
    "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
    "Relative speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
    "Its perigee (km)": st.column_config.NumberColumn(format="%.0f"),
    "Its apogee (km)": st.column_config.NumberColumn(format="%.0f"),
}


with tab_triage:
    st.subheader("Conjunctions & triage")
    asset_ids = ids  # active satellites only (from active.json)
    with st.form("tri_asset_form"):
        a1, a2, a3 = st.columns([3, 1, 1])
        asset = a1.selectbox("Object to screen", asset_ids,
                             index=asset_ids.index(25544) if 25544 in asset_ids else 0, format_func=label)
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
        st.caption(f"{records[res['asset']]['OBJECT_NAME']} · window from {res['start']} UTC for {res['hours'] / 24:g} days. "
                   "Objects holding within ~20 km of each other the whole window (docked/formation) are ignored."
                   + (f" The object's orbit ends after {res['hours'] / 24:.1f} days, so only that part was screened."
                      if res["hours"] < res["asked"] - 1 else ""))
        if not len(ev):
            st.success("No close approaches against any catalogue object inside the threshold in this window.")
        else:
            st.dataframe(events_table(ev), width="stretch", hide_index=True, column_config=EVENT_CONFIG)

# floating chat launcher (top-right, below Streamlit's toolbar); the popover opens downward and closes with the same button
st.markdown(
    """<style>
    .st-key-ai_fab { position: fixed; top: 3.75rem; right: 1.5rem; z-index: 1000; width: auto !important; }
    .st-key-ai_fab button { border-radius: 999px; box-shadow: 0 2px 10px rgba(0,0,0,.35); font-weight: 600; }
    [data-testid="stPopoverBody"] { width: min(520px, 92vw); max-height: calc(100vh - 8rem); overflow-y: auto; }
    </style>""",
    unsafe_allow_html=True,
)
with st.container(key="ai_fab"):
    with st.popover("💬 Ask the assistant"):
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

                t_chat, t_brief, t_risk = st.tabs(["Ask the data", "Object briefing", "Risk explanation"])
                for tab, kind, task, button in ((t_brief, "brief", llm.BRIEFING_TASK, "Generate briefing"),
                                                (t_risk, "risk", llm.RISK_TASK, "Explain risk score")):
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
                    st.caption("Examples: *Which objects below 200 km have an inclination over 50°?* · "
                               "*Why is STARLINK-1433 ranked so high?* · *Any close approaches for the ISS in the next 6 hours?*")
                    chat_hist = st.session_state.setdefault("ai_chat", [])
                    for m in chat_hist:
                        with st.chat_message(m["role"]):
                            st.markdown(m["content"])
                            if m.get("trace"):
                                st.caption("Looked up: " + ", ".join(f"{t['tool']}({t['input']})" for t in m["trace"]))
                    q = st.chat_input("Ask about the catalogue…")
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
        "(so 50 listed satellites show 10 / 30 / 10, and a list of 10 is all red). Debris is not on the watchlist; it can be drawn on the globe.\n"
        "- **Footprint:** latitude band ±min(i, 180−i); time per latitude from sin φ = sin i · sin u.\n"
        "- **Conjunctions:** coarse-grid SGP4 screening with 0.5 s refinement of local minima, against every satellite and debris fragment.\n"
        "- **Debris:** the debris groups (TLE or CelesTrak JSON files in data/debris) are parsed into the same record format. They are not on the watchlist "
        "or in the pickers; they are only drawn as orbits on the Watchlist globe and used as close-approach partners when an active satellite is screened "
        "(Conjunctions & Triage, and the close-approach terms of the risk score). Altitude and risk figures are for active satellites only."
        "Altitude bands are the densest ranges of the active catalogue, found from the data.\n"
        "- **Assistant (optional):** off unless an API key is supplied. The model only receives numbers computed above, or calls the app's own lookup functions; it never computes orbits.\n"
        "- **Limits:** one element set per object (no history until snapshots are added), mean elements are only km-accurate, "
        "and active satellites that manoeuvre invalidate drag-based decay estimates."
    )
