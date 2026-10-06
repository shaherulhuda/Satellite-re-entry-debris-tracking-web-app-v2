"""Re-Entry & Debris Mitigation Assistant (Streamlit UI). Physics lives in the other modules."""
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sgp4.api import jday

import charts
import conjunction
import debris
import decay
import footprint
import globe
import history
import llm
import maneuver
import orbits
import triage

st.set_page_config(page_title="Re-Entry & Debris Mitigation Assistant", layout="wide")


@st.cache_data
def get_data():
    """Active satellites + the Fengyun-1C debris set, one record per NORAD id."""
    active, deb = orbits.load_records(), debris.load_debris_records()
    recs = debris.merge_records(active, deb)
    cat = decay.estimate_decay(orbits.catalog_from_records(recs))
    cat["type"] = cat["OBJECT_NAME"].map(decay.classify_type)
    deb_ids = {r["NORAD_CAT_ID"] for r in deb}
    cat["source"] = np.where(cat["NORAD_CAT_ID"].isin(deb_ids), "debris", "active")
    return cat, {r["NORAD_CAT_ID"]: r for r in recs}


@st.cache_data
def get_history():
    return history.load_history()


df, records = get_data()
hist_all = get_history()
components_df = decay.risk_components(df)

st.title("Re-Entry & Debris Mitigation Assistant")
n_deb = int((df["source"] == "debris").sum())
st.caption(f"Data snapshot from {orbits.snapshot_date(df)} · {len(df):,} objects"
           + (f" (including {n_deb:,} Fengyun-1C debris fragments)" if n_deb else ""))

tab_watch, tab_track, tab_conj, tab_triage = st.tabs(["Watchlist", "Tracker", "Conjunctions", "Triage desk"])


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

# ---------------------------------------------------------------- Assistant (optional, needs an API key)
# ---------------------------------------------------------------- Triage desk
ctx = llm.DataContext(df, records, components_df, weights, hist_all)
fmt_utc = lambda jd: conjunction.jd_to_datetime(jd).strftime("%Y-%m-%d %H:%M:%S")


def events_table(ev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Object": ev["name"], "NORAD ID": ev["norad_id"],
        "Closest approach (UTC)": [fmt_utc(j) for j in ev["tca_jd"]],
        "Miss distance (km)": ev["miss_km"], "Relative speed (km/s)": ev["rel_speed_km_s"],
        "Its perigee (km)": ev["other_perigee_km"], "Its apogee (km)": ev["other_apogee_km"],
    })


EVENT_CONFIG = {
    "NORAD ID": st.column_config.NumberColumn(format="%d"),
    "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
    "Relative speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
    "Its perigee (km)": st.column_config.NumberColumn(format="%.0f"),
    "Its apogee (km)": st.column_config.NumberColumn(format="%.0f"),
}


def memo_block(kind: str, facts: dict, task: str):
    """Button + saved result for one LLM decision brief; the facts it is built from are always shown."""
    ready, _, msg = ai_status()
    if st.button("Write decision memo (AI)", key=f"memo_btn_{kind}", disabled=not ready):
        with st.spinner("Asking the model…"):
            res = ai_guarded(lambda c: llm.explain(c, st.session_state.get("ai_model", llm.DEFAULT_MODEL), facts, task))
        if res:
            st.session_state[f"memo_{kind}"] = res
    if not ready:
        st.caption(msg)
    saved = st.session_state.get(f"memo_{kind}")
    if saved:
        st.markdown(saved[0])
        st.caption(f"{saved[1]['input_tokens']} input / {saved[1]['output_tokens']} output tokens. "
                   "The model only wrote prose around the numbers below; it did not compute them.")
    with st.expander("Facts the memo is built from (computed by this app)"):
        st.json(facts)


with tab_triage:
    st.subheader("Triage desk")
    n_deb_t = int((df["source"] == "debris").sum())
    st.caption(f"Debris set: {n_deb_t:,} Fengyun-1C breakup fragments from the bundled TLE file. "
               "Screening and cost figures are first-order estimates from mean elements, for triage, not operational decisions.")
    mode = st.radio("View", ["Protect a satellite", "Follow a debris object"], horizontal=True, key="tri_mode")

    if mode == "Protect a satellite":
        asset_ids = df[df["source"] == "active"].sort_values("OBJECT_NAME")["NORAD_CAT_ID"].astype(int).tolist()
        with st.form("tri_asset_form"):
            a1, a2, a3, a4 = st.columns([3, 1, 1, 2])
            asset = a1.selectbox("Satellite to protect", asset_ids,
                                 index=asset_ids.index(25544) if 25544 in asset_ids else 0, format_func=label)
            hours = a2.slider("Window (h)", 6, 72, 48)
            thr = a3.slider("Report within (km)", 5, 100, 25)
            start_mode = a4.radio("Window starts at", ["Snapshot time", "Now"], horizontal=True)
            run = st.form_submit_button("Screen against debris")
        if run:
            jd0, ts0 = start_time(start_mode)
            with st.spinner("Propagating debris…"):
                try:
                    ev, n = triage.screen_asset_vs_debris(records[int(asset)], df, records, jd0, hours=hours, threshold_km=thr)
                except Exception as exc:
                    st.error(f"Screening failed: {exc}")
                else:
                    st.session_state["tri_asset"] = {"asset": int(asset), "events": ev, "n": n, "hours": hours, "thr": thr,
                                                     "start": ts0.strftime("%Y-%m-%d %H:%M")}
                    st.session_state.pop("memo_asset", None)
        res = st.session_state.get("tri_asset")
        if res:
            arow = df[df["NORAD_CAT_ID"] == res["asset"]].iloc[0]
            ev = res["events"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Debris in a similar altitude band", f"{res['n']:,}")
            m2.metric(f"Passes within {res['thr']} km", f"{len(ev)}")
            m3.metric("Closest pass", f"{ev['miss_km'].min():.1f} km" if len(ev) else "none")
            st.caption(f"{records[res['asset']]['OBJECT_NAME']} · window from {res['start']} UTC for {res['hours']} h. "
                       "Objects holding within ~20 km of each other the whole window (docked/formation) are ignored.")
            if not len(ev):
                st.success("No debris passes inside the threshold in this window.")
            else:
                st.dataframe(events_table(ev), width="stretch", hide_index=True, column_config=EVENT_CONFIG)

                st.markdown("#### Avoidance cost")
                st.caption("Delta-v from a small tangential burn (Clohessy-Wiltshire) and fuel from the rocket equation. "
                           "The existing miss direction is unknown, so a best case (burn moves it away) and a worst case are shown.")
                labels = [f"{r.name} · {r.miss_km:.1f} km at {fmt_utc(r.tca_jd)}" for r in ev.itertuples()]
                pick = st.selectbox("Pass to plan around", range(len(ev)), format_func=lambda i: labels[i], key="tri_pass")
                p1, p2, p3, p4 = st.columns(4)
                preset = p1.selectbox("Vehicle", list(maneuver.PRESETS), key="tri_preset")
                m0, isp0 = maneuver.PRESETS[preset]
                mass = p2.number_input("Mass (kg)", 1.0, 1_000_000.0, float(m0), key=f"tri_mass_{preset}")
                isp = p3.number_input("Isp (s)", 50.0, 5000.0, float(isp0), key=f"tri_isp_{preset}")
                target = p4.number_input("Target miss (km)", 1.0, 100.0, 5.0, key="tri_target")
                chosen = ev.iloc[int(pick)]
                plan = maneuver.avoidance_plan(float(chosen["miss_km"]), target, float(arow["MEAN_MOTION"]), mass, isp)
                if plan.empty:
                    st.success(f"This pass already misses by {chosen['miss_km']:.1f} km, above the {target:.0f} km target: no burn needed.")
                else:
                    st.dataframe(pd.DataFrame({
                        "Burn lead time (h)": plan["lead_h"], "Delta-v best (m/s)": plan["dv_best_ms"],
                        "Delta-v worst (m/s)": plan["dv_worst_ms"], "Propellant worst (kg)": plan["fuel_worst_kg"],
                        "% of vehicle mass (worst)": plan["fuel_worst_pct_mass"]}),
                        width="stretch", hide_index=True, column_config={
                            "Delta-v best (m/s)": st.column_config.NumberColumn(format="%.3f"),
                            "Delta-v worst (m/s)": st.column_config.NumberColumn(format="%.3f"),
                            "Propellant worst (kg)": st.column_config.NumberColumn(format="%.4f"),
                            "% of vehicle mass (worst)": st.column_config.NumberColumn(format="%.4f")})
                    st.caption("Burning earlier is far cheaper: the displacement grows with lead time. "
                               "Electric thrusters burn for hours, so treat these as lower bounds for them.")

                st.markdown("#### Decision memo")
                chosen_info = {"object": chosen["name"], "norad_id": int(chosen["norad_id"]),
                               "closest_approach_utc": fmt_utc(chosen["tca_jd"]), "miss_km": round(float(chosen["miss_km"]), 2),
                               "relative_speed_km_s": round(float(chosen["rel_speed_km_s"]), 2)}
                inputs = {"vehicle": preset, "mass_kg": mass, "isp_s": isp, "target_miss_km": target}
                facts = triage.asset_facts(llm.object_facts(ctx, res["asset"], include_position=False), ev, res["n"],
                                           res["hours"], res["thr"], res["start"], plan, inputs, chosen_info)
                memo_block("asset", facts, llm.DECISION_MEMO_TASK)

    else:
        deb = df[df["source"] == "debris"].sort_values("decay_days", na_position="last")
        deb_ids = deb["NORAD_CAT_ID"].astype(int).tolist()
        info = {int(r.NORAD_CAT_ID): r for r in deb.itertuples()}

        def deb_label(i):
            r = info[i]
            return f"{r.OBJECT_NAME} ({i}) · est. {decay.format_days(r.decay_days)} · perigee {r.perigee_km:.0f} km"

        dsel = st.selectbox("Debris object (soonest estimated re-entry first)", deb_ids, format_func=deb_label, key="tri_deb")
        drow = df[df["NORAD_CAT_ID"] == dsel].iloc[0]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Perigee / apogee", f"{drow['perigee_km']:.0f} / {drow['apogee_km']:.0f} km")
        e2.metric("Inclination", f"{drow['INCLINATION']:.1f}°")
        e3.metric("Est. time to re-entry", decay.format_days(drow["decay_days"]))
        e4.metric("Rough range", decay.format_range(drow["decay_days_low"], drow["decay_days_high"]))
        st.caption(f"Decay status: **{drow['decay_status']}**. Rough estimate (factor 2 or worse).")

        bands = triage.altitude_bands(df)
        h0 = float(decay.mean_altitude_km(drow["MEAN_MOTION"]))
        crossings = triage.band_crossings(h0, float(drow["decay_km_day"]), bands)
        st.markdown("#### 1. Sinking through the crowded altitudes")
        if np.isfinite(drow["decay_km_day"]) and drow["decay_km_day"] > 0:
            st.plotly_chart(charts.eol_figure(h0, float(drow["decay_km_day"]), bands), width="stretch")
            if len(crossings):
                st.dataframe(pd.DataFrame({
                    "Altitude band": crossings["band"], "Active objects there": crossings["n_objects"],
                    "Enters in": ["already inside" if i else decay.format_days(d) for i, d in
                                  zip(crossings["already_inside"], crossings["enters_days"])],
                    "Leaves in": crossings["leaves_days"].map(decay.format_days)}), width="stretch", hide_index=True)
            st.caption("Bands are the altitude ranges where the active catalogue is densest (found from the data), plus the ISS orbit.")
        else:
            st.info("No measurable decay: the model cannot project this object's descent (it may stay in orbit for decades).")

        st.markdown("#### 2. Who it passes close to")
        with st.form("tri_deb_form"):
            b1, b2, b3 = st.columns([1, 1, 2])
            dh = b1.slider("Window (h)", 6, 48, 24)
            dthr = b2.slider("Report within (km)", 5, 100, 25)
            dstart = b3.radio("Window starts at", ["Snapshot time", "Now"], horizontal=True)
            drun = st.form_submit_button("Screen against active satellites")
        if drun:
            jd0, ts0 = start_time(dstart)
            with st.spinner("Propagating satellites… (can take ~10-20 s)"):
                try:
                    thr_ev, thr_n = triage.screen_debris_vs_satellites(records[int(dsel)], df, records, jd0, hours=dh, threshold_km=dthr)
                except Exception as exc:
                    st.error(f"Screening failed: {exc}")
                else:
                    st.session_state["tri_threat"] = {"deb": int(dsel), "events": thr_ev, "n": thr_n, "hours": dh,
                                                      "thr": dthr, "start": ts0.strftime("%Y-%m-%d %H:%M")}
                    st.session_state.pop("memo_eol", None)
        tres = st.session_state.get("tri_threat")
        if tres and tres["deb"] == int(dsel):
            st.write(f"{tres['n']:,} satellites in a similar altitude band · **{len(tres['events'])}** passes within {tres['thr']} km "
                     f"in the {tres['hours']} h from {tres['start']} UTC")
            if len(tres["events"]):
                st.dataframe(events_table(tres["events"]), width="stretch", hide_index=True, column_config=EVENT_CONFIG)

        st.markdown("#### 3. Where it can come down")
        st.plotly_chart(charts.footprint_figure(float(drow["INCLINATION"])), width="stretch")
        st.caption("Shows the latitudes this orbit can reach, not a landing prediction.")

        st.markdown("#### Decision memo")
        threats = tres["events"] if tres and tres["deb"] == int(dsel) else pd.DataFrame(columns=triage.EVENT_COLUMNS)
        facts = triage.eol_facts(llm.object_facts(ctx, int(dsel), include_position=False), crossings, threats,
                                 tres["n"] if tres and tres["deb"] == int(dsel) else 0,
                                 tres["hours"] if tres else 0, tres["thr"] if tres else 0,
                                 tres["start"] if tres else "not screened", bands)
        if not tres or tres["deb"] != int(dsel):
            st.caption("Run the satellite screening above first so the memo can mention close passes.")
        memo_block("eol", facts, llm.EOL_MEMO_TASK)

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
        "- **Risk score:** weighted blend of perigee, decay time, inclination and object type; weights are adjustable above.\n"
        "- **Footprint:** latitude band ±min(i, 180−i); time per latitude from sin φ = sin i · sin u.\n"
        "- **Conjunctions:** coarse-grid SGP4 screening with 0.5 s refinement of local minima.\n"
        "- **Debris and triage:** the Fengyun-1C TLE file is parsed into the same record format and screened with the same SGP4 conjunction code. "
        "Avoidance cost: tangential burn displacement from Clohessy-Wiltshire (best/worst case for the unknown miss direction), propellant from the rocket equation. "
        "Altitude bands are the densest ranges of the active catalogue, found from the data.\n"
        "- **Assistant (optional):** off unless an API key is supplied. The model only receives numbers computed above, or calls the app's own lookup functions; it never computes orbits.\n"
        "- **Limits:** one element set per object (no history until snapshots are added), mean elements are only km-accurate, "
        "and active satellites that manoeuvre invalidate drag-based decay estimates."
    )
