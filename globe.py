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


def orbit_figure(paths: list[dict], height: int = 650, highlight: bool = False) -> go.Figure:
    """paths: dicts with keys name, x, y, z, now, and optional hover (text)."""
    fig = go.Figure(_earth_traces())
    for p in paths:
        color = "#ff5a5f" if highlight else None
        fig.add_trace(go.Scatter3d(
            x=p["x"], y=p["y"], z=p["z"], mode="lines", name=p["name"],
            line=dict(width=4 if highlight else 2, color=color), hovertext=p.get("hover", p["name"]),
            hoverinfo="text", showlegend=highlight,
        ))
        nx, ny, nz = p["now"]
        fig.add_trace(go.Scatter3d(
            x=[nx], y=[ny], z=[nz], mode="markers", name=f"{p['name']} (now)",
            marker=dict(size=6 if highlight else 3, color="#ffd23f"),
            hovertext=p.get("hover", p["name"]), hoverinfo="text", showlegend=False,
        ))
    fig.update_layout(**_layout(height))
    return fig
