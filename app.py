"""Re-Entry & Debris Mitigation Assistant (Streamlit UI)."""
import streamlit.components.v1 as components
import streamlit as st

import globe
import orbits

st.set_page_config(page_title="Re-Entry & Debris Mitigation Assistant", layout="wide")


@st.cache_data
def get_catalog():
    return orbits.load_catalog()


@st.cache_data
def get_records_by_norad():
    return {r["NORAD_CAT_ID"]: r for r in orbits.load_records()}


df = get_catalog()
records = get_records_by_norad()

st.title("Re-Entry & Debris Mitigation Assistant")
st.caption(f"Data snapshot from {orbits.snapshot_date(df)} · {len(df):,} objects")

tab_watch, tab_track = st.tabs(["Watchlist", "Tracker"])

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


with tab_watch:
    st.subheader("50 lowest-perigee objects")
    wl = orbits.watchlist(df, 50)
    st.caption("Orbits of the watchlist objects around a rotating Earth (one revolution each; yellow dots = positions now). "
               "The animation loops until you press Pause. Drag to rotate, scroll to zoom.")
    components.html(globe.orbit_html(watchlist_paths(tuple(int(i) for i in wl["NORAD ID"]))), height=660)
    st.dataframe(
        wl,
        width='stretch',
        hide_index=True,
        column_config={
            "NORAD ID": st.column_config.NumberColumn(format="%d"),
            "Perigee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Apogee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Inclination (°)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

with tab_track:
    names = sorted(df["OBJECT_NAME"].unique())
    name = st.selectbox("Object", names, index=names.index("ISS (ZARYA)") if "ISS (ZARYA)" in names else 0)
    row = df[df["OBJECT_NAME"] == name].iloc[0]
    record = records[int(row["NORAD_CAT_ID"])]

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
