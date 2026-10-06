"""Re-Entry & Debris Mitigation Assistant (Streamlit UI). Physics lives in the other modules."""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sgp4.api import jday

import charts
import conjunction
import decay
import footprint
import globe
import history
import orbits

st.set_page_config(page_title="Re-Entry & Debris Mitigation Assistant", layout="wide")


@st.cache_data
def get_catalog():
    cat = decay.estimate_decay(orbits.load_catalog())
    cat["type"] = cat["OBJECT_NAME"].map(decay.classify_type)
    return cat


@st.cache_data
def get_records_by_norad():
    return {r["NORAD_CAT_ID"]: r for r in orbits.load_records()}


@st.cache_data
def get_history():
    return history.load_history()


df = get_catalog()
records = get_records_by_norad()
hist_all = get_history()
components_df = decay.risk_components(df)

st.title("Re-Entry & Debris Mitigation Assistant")
st.caption(f"Data snapshot from {orbits.snapshot_date(df)} · {len(df):,} objects")

tab_watch, tab_track, tab_conj = st.tabs(["Watchlist", "Tracker", "Conjunctions"])


@st.cache_data(ttl=300)
def watchlist_paths(norad_ids: tuple):
    out = []
    for nid in norad_ids:
        rec = records[nid]
        try:
            p = orbits.orbit_path_xyz(rec, n_points=120)
        except Exception:
            continue
        row = df[df["NORAD_CAT_ID"] == nid].iloc[0]
        p["name"] = rec["OBJECT_NAME"]
        p["hover"] = f"{rec['OBJECT_NAME']}<br>perigee {row['perigee_km']:.0f} km · apogee {row['apogee_km']:.0f} km"
        out.append(p)
    return out


def label(norad_id: int) -> str:
    r = records[norad_id]
    return f"{r['OBJECT_NAME']} ({norad_id})"


# ---------------------------------------------------------------- Watchlist
with tab_watch:
    st.subheader("Watchlist")
    c_mode, c_n = st.columns([3, 1])
    mode = c_mode.radio("Rank by", ["Risk score (highest first)", "Soonest estimated re-entry", "Lowest perigee"],
                        horizontal=True)
    n_show = c_n.number_input("Objects", 10, 200, 50, step=10)

    with st.expander("Risk score: formula and weights"):
        st.markdown(
            "**risk = 100 × (w₁·perigee + w₂·decay + w₃·inclination + w₄·type) / Σw**, each term scaled 0–1:\n"
            f"- **perigee** = (700 − perigee km) / 550, clipped (150 km → 1, ≥ 700 km → 0)\n"
            f"- **decay** = how soon re-entry is estimated: 1 at ≤ 10 days, 0 at ≥ 10 years (log scale); 0 if no measurable decay\n"
            "- **inclination** = share of Earth's surface the orbit can overfly, sin(max latitude)\n"
            "- **type** = debris 1.0, rocket body 0.8, payload 0.3 (guessed from the catalogue name; "
            "uncontrolled objects cannot be steered away)\n\n"
            "Change the weights below; the ranking updates."
        )
        w1, w2, w3, w4 = st.columns(4)
        weights = {
            "perigee": w1.slider("Perigee", 0.0, 1.0, decay.DEFAULT_WEIGHTS["perigee"], 0.05),
            "decay": w2.slider("Decay", 0.0, 1.0, decay.DEFAULT_WEIGHTS["decay"], 0.05),
            "inclination": w3.slider("Inclination", 0.0, 1.0, decay.DEFAULT_WEIGHTS["inclination"], 0.05),
            "type": w4.slider("Type", 0.0, 1.0, decay.DEFAULT_WEIGHTS["type"], 0.05),
        }

    ranked = df.copy()
    ranked["risk"] = decay.risk_score(components_df, weights)
    if mode.startswith("Risk"):
        ranked = ranked.sort_values("risk", ascending=False)
    elif mode.startswith("Soonest"):
        ranked = ranked.sort_values("decay_days", na_position="last")
    else:
        ranked = ranked.sort_values("perigee_km")
    top = ranked.head(int(n_show))

    table = pd.DataFrame({
        "Name": top["OBJECT_NAME"], "NORAD ID": top["NORAD_CAT_ID"], "Type": top["type"],
        "Perigee (km)": top["perigee_km"], "Apogee (km)": top["apogee_km"], "Inclination (°)": top["INCLINATION"],
        "Est. time to re-entry": top["decay_days"].map(decay.format_days),
        "Range (±×2)": [decay.format_range(lo, hi) for lo, hi in zip(top["decay_days_low"], top["decay_days_high"])],
        "Risk score": top["risk"],
    })
    st.caption("Decay figures are rough estimates from each object's measured drag (see *How these numbers are computed*). "
               "Many active satellites (e.g. Starlink) manoeuvre to hold altitude, so their drag-based estimate is not a prediction.")
    st.caption("Orbits of the listed objects around a rotating Earth (one revolution each; yellow dots = positions now). "
               "The animation loops until you press Pause. Drag to rotate, scroll to zoom.")
    components.html(globe.orbit_html(watchlist_paths(tuple(int(i) for i in table["NORAD ID"]))), height=660)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "NORAD ID": st.column_config.NumberColumn(format="%d"),
            "Perigee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Apogee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Inclination (°)": st.column_config.NumberColumn(format="%.2f"),
            "Risk score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
        },
    )

# ---------------------------------------------------------------- Tracker
with tab_track:
    ids = df.sort_values("OBJECT_NAME")["NORAD_CAT_ID"].astype(int).tolist()
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

# ---------------------------------------------------------------- Conjunctions
with tab_conj:
    st.subheader("Close-approach screening")
    st.caption("Screens one object against every catalogue object in a similar altitude band, using SGP4 on the stored elements. "
               "This is a screening aid, not an operational conjunction assessment: no uncertainty or collision probability.")
    snap = df["EPOCH_DT"].max()
    with st.form("conj"):
        a, b, c, d = st.columns([3, 1, 1, 2])
        cn = a.selectbox("Primary object", ids, index=default, format_func=label)
        hours = b.slider("Window (h)", 1, 24, 6)
        thr = c.slider("Report within (km)", 1, 50, 10)
        start_mode = d.radio("Window starts at", ["Snapshot time", "Now"], horizontal=True)
        go_btn = st.form_submit_button("Run screening")
    if go_btn:
        if start_mode == "Now":
            n = pd.Timestamp.utcnow()
        else:
            n = snap
        jd, fr = jday(n.year, n.month, n.day, n.hour, n.minute, n.second + n.microsecond / 1e6)
        with st.spinner("Propagating candidates…"):
            try:
                events, n_cand = conjunction.screen(records[int(cn)], df, records, jd + fr, hours=hours, threshold_km=thr)
            except Exception as exc:
                st.error(f"Screening failed: {exc}")
            else:
                st.write(f"Window start: {n:%Y-%m-%d %H:%M} UTC · {n_cand} objects in a similar altitude band · "
                         f"**{len(events)}** approaches within {thr} km")
                if len(events):
                    ev = pd.DataFrame({
                        "Object": events["name"], "NORAD ID": events["norad_id"],
                        "Closest approach (UTC)": [conjunction.jd_to_datetime(j).strftime("%Y-%m-%d %H:%M:%S") for j in events["tca_jd"]],
                        "Miss distance (km)": events["miss_km"], "Relative speed (km/s)": events["rel_speed_km_s"],
                    })
                    st.dataframe(ev, width="stretch", hide_index=True, column_config={
                        "NORAD ID": st.column_config.NumberColumn(format="%d"),
                        "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
                        "Relative speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
                    })
                else:
                    st.info("No approaches inside the threshold in this window.")
                st.caption("Objects that stay within ~20 km of the primary for the whole window (docked modules, tight formations) are ignored. "
                           "Mean-element positions are only good to roughly a kilometre or more, and worse for old element sets.")

with st.expander("How these numbers are computed"):
    st.markdown(
        "- **Perigee / apogee:** a from mean motion via Kepler's third law (n²a³ = μ); altitude = a(1∓e) − 6378.137 km.\n"
        "- **Positions and orbits:** SGP4 (via skyfield) on the stored mean elements; the globe is drawn in Earth-fixed axes with the Earth rotating at its real rate.\n"
        "- **Decay estimate:** the measured `MEAN_MOTION_DOT` (×2, since the field is ṅ/2) gives da/dt = −(2/3)·a·ṅ/n. "
        "That rate is scaled by air density (exponential atmosphere table) and integrated down to 120 km. "
        "Requires positive `BSTAR` (drag term). Uncertainty is at least a factor of 2.\n"
        "- **Risk score:** weighted blend of perigee, decay time, inclination and object type; weights are adjustable above.\n"
        "- **Footprint:** latitude band ±min(i, 180−i); time per latitude from sin φ = sin i · sin u.\n"
        "- **Conjunctions:** coarse-grid SGP4 screening with 0.5 s refinement of local minima.\n"
        "- **Limits:** one element set per object (no history until snapshots are added), mean elements are only km-accurate, "
        "and active satellites that manoeuvre invalidate drag-based decay estimates."
    )
