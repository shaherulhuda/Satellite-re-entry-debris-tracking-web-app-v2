"""3D Earth + orbit figures (plotly Scatter3d). Offline: Earth is drawn procedurally."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from orbits import EARTH_RADIUS_KM as R


def _earth_traces() -> list:
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    sphere = go.Surface(
        x=x, y=y, z=z, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#0b2a4a"], [0.5, "#1c5d99"], [1, "#4aa3df"]],
        surfacecolor=z, lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2),
    )
    traces = [sphere]
    # graticule: meridians every 30 deg, parallels every 30 deg (slightly above surface)
    r = R * 1.002
    t = np.linspace(0, 2 * np.pi, 120)
    grid = dict(mode="lines", line=dict(color="rgba(200,220,255,0.35)", width=1),
                hoverinfo="skip", showlegend=False)
    for lon in np.deg2rad(np.arange(0, 180, 30)):
        traces.append(go.Scatter3d(
            x=r * np.cos(t) * np.cos(lon), y=r * np.cos(t) * np.sin(lon), z=r * np.sin(t), **grid))
    for lat in np.deg2rad(np.arange(-60, 61, 30)):
        traces.append(go.Scatter3d(
            x=r * np.cos(lat) * np.cos(t), y=r * np.cos(lat) * np.sin(t),
            z=np.full_like(t, r * np.sin(lat)), **grid))
    return traces


def _layout(height: int) -> dict:
    ax = dict(visible=False, showbackground=False)
    return dict(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#05070d", font=dict(color="#dde6f5"),
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor="#05070d",
                   camera=dict(eye=dict(x=1.2, y=0.9, z=0.6))),
        legend=dict(orientation="h", y=0.02, bgcolor="rgba(0,0,0,0)"),
    )


def orbit_figure(paths: list[dict], height: int = 650, highlight: bool = False,
                 spin: bool = True, step_ms: int = 80) -> go.Figure:
    """paths: dicts with keys name, x, y, z, now, optional hover (text) and anim.

    If every path has `anim` (3 x F positions at 1-minute steps), the figure gets
    Play/Pause buttons and a slider that move all satellites along their orbits,
    and (with spin=True) orbit the camera once around the Earth.
    """
    fig = go.Figure(_earth_traces())
    for p in paths:
        fig.add_trace(go.Scatter3d(
            x=p["x"], y=p["y"], z=p["z"], mode="lines", name=p["name"],
            line=dict(width=4 if highlight else 2, color="#ff5a5f" if highlight else None),
            hovertext=p.get("hover", p["name"]), hoverinfo="text", showlegend=highlight,
        ))
    pts = np.array([p["now"] for p in paths]).T
    fig.add_trace(go.Scatter3d(
        x=pts[0], y=pts[1], z=pts[2], mode="markers", name="Position",
        marker=dict(size=6 if highlight else 3, color="#ffd23f"),
        hovertext=[p.get("hover", p["name"]) for p in paths], hoverinfo="text", showlegend=False,
    ))
    marker_idx = len(fig.data) - 1
    fig.update_layout(**_layout(height))

    if all("anim" in p for p in paths):
        n = min(p["anim"].shape[1] for p in paths)
        cam0 = fig.layout.scene.camera.eye
        r0, z0 = float(np.hypot(cam0.x, cam0.y)), float(cam0.z)
        phi0 = float(np.arctan2(cam0.y, cam0.x))
        frames = []
        for k in range(n):
            frame = dict(
                name=str(k), traces=[marker_idx],
                data=[go.Scatter3d(x=[p["anim"][0, k] for p in paths],
                                   y=[p["anim"][1, k] for p in paths],
                                   z=[p["anim"][2, k] for p in paths])],
            )
            if spin:
                phi = phi0 + 2 * np.pi * k / (n - 1)
                frame["layout"] = dict(scene=dict(camera=dict(
                    eye=dict(x=r0 * np.cos(phi), y=r0 * np.sin(phi), z=z0))))
            frames.append(go.Frame(**frame))
        fig.frames = frames
        play = dict(frame=dict(duration=step_ms, redraw=True), fromcurrent=True,
                    transition=dict(duration=0), mode="immediate")
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left", active=-1, bordercolor="#5b7bb0", x=0.02, y=0.98, xanchor="left", yanchor="top",
                bgcolor="#1c2a44", font=dict(color="#dde6f5"),
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, play]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ])],
            sliders=[dict(
                x=0.1, y=0.02, len=0.85, currentvalue=dict(prefix="Time: +", suffix=" min"),
                font=dict(color="#dde6f5"),
                steps=[dict(label=str(k), method="animate",
                            args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                       for k in range(n)],
            )],
        )
    return fig
