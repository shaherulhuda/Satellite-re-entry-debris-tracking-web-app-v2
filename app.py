"""Re-Entry & Debris Mitigation Assistant (Streamlit UI)."""
import plotly.graph_objects as go
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

@st.cache_data
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
    st.caption("Orbits of the watchlist objects around Earth (inertial frame, one revolution each; yellow dots = positions now). Drag to rotate, scroll to zoom.")
    st.plotly_chart(globe.orbit_figure(watchlist_paths(tuple(int(i) for i in wl["NORAD ID"]))),
                    width="stretch")
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

        st.markdown("**3D orbit** (inertial frame, one revolution; yellow dot = position now)")
        path = orbits.orbit_path_xyz(record)
        path["name"] = name
        st.plotly_chart(globe.orbit_figure([path], height=600, highlight=True), width="stretch")

        st.markdown("**Ground track** (next 3 orbits)")
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(lat=track["lat"], lon=track["lon"], mode="lines",
                                    line=dict(width=2, color="#1f77b4"), name="3-orbit ground track"))
        fig.add_trace(go.Scattergeo(lat=[pos["lat"]], lon=[pos["lon"]], mode="markers",
                                    marker=dict(size=11, color="red"), name="Current position"))
        fig.update_geos(projection_type="equirectangular", showland=True, showcountries=True,
                        lataxis_range=[-90, 90], lonaxis_range=[-180, 180])
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width='stretch')
