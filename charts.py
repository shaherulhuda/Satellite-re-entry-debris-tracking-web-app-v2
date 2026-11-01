"""2D plotly figures for decay, perigee history and the latitude footprint (offline)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import decay
import footprint
from globe import coastline_trace_2d, float_on_dark, lock_for_touch


def decay_figure(h0_km: float, rate_km_day: float, hist=None, height: int = 360, phone: bool = False):
    """Perigee history (points) plus the model's projected altitude decline."""
    fig = go.Figure()
    proj = decay.altitude_projection(h0_km, rate_km_day)
    if proj is not None:
        t, h = proj
        fig.add_trace(go.Scatter(x=t, y=h, mode="lines", name="Model projection (mean altitude)",
                                 line=dict(color="#d62728", dash="dot")))
        fig.add_hline(y=decay.REENTRY_ALT_KM, line=dict(color="#888", dash="dash"),
                      annotation_text=f"re-entry threshold {decay.REENTRY_ALT_KM:.0f} km")
        fig.update_xaxes(title="Days from the element epoch")
    if hist is not None and len(hist):
        t0 = hist["epoch"].iloc[0]
        x = (hist["epoch"] - t0).dt.total_seconds() / 86400.0
        fig.add_trace(go.Scatter(x=x, y=hist["perigee_km"], mode="markers+lines", name="Recorded perigee",
                                 marker=dict(size=9, color="#1f77b4")))
    fig.update_yaxes(title="Altitude (km)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25))
    if phone:  # narrow screen: taller plot, smaller type, short axis titles, legend stacked underneath
        fig.update_xaxes(title="Days from epoch")
        fig.update_yaxes(title="Altitude (km)", title_standoff=6, automargin=True)
        fig.update_layout(height=420, margin=dict(l=0, r=4, t=10, b=0), font=dict(size=11),
                          legend=dict(orientation="h", x=0, y=-0.3, yanchor="top"))
        lock_for_touch(fig)
    return float_on_dark(fig)


def footprint_figure(inclination_deg: float, height: int = 430, phone: bool = False):
    """World map with the overflown latitude band, next to the share of time spent per latitude."""
    phi = footprint.max_latitude(inclination_deg)
    lat, p = footprint.latitude_distribution(inclination_deg)
    if phone:
        return _footprint_phone(phi, lat, p)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.72, 0.28],
                        horizontal_spacing=0.02)
    fig.add_trace(coastline_trace_2d(), row=1, col=1)
    fig.add_shape(type="rect", x0=-180, x1=180, y0=-phi, y1=phi, fillcolor="rgba(214,39,40,0.18)",
                  line=dict(color="rgba(214,39,40,0.8)", width=1), row=1, col=1)
    fig.add_trace(go.Bar(y=lat, x=p * 100, orientation="h", marker_color="#d62728",
                         hovertemplate="%{y:.0f}°: %{x:.2f}% of time<extra></extra>", showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(range=[-180, 180], title="Longitude (°)", dtick=60, row=1, col=1)
    fig.update_xaxes(title="% of time per 2° band", row=1, col=2)
    fig.update_yaxes(range=[-90, 90], title="Latitude (°)", dtick=30, row=1, col=1)
    fig.update_yaxes(range=[-90, 90], dtick=30, row=1, col=2)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), bargap=0.05)
    return float_on_dark(fig)


def _footprint_phone(phi, lat, p) -> go.Figure:
    """Narrow screens: the map on top at full width, the time-per-latitude bars underneath (instead of beside it)."""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.16)
    fig.add_trace(coastline_trace_2d(), row=1, col=1)
    fig.add_shape(type="rect", x0=-180, x1=180, y0=-phi, y1=phi, fillcolor="rgba(214,39,40,0.18)",
                  line=dict(color="rgba(214,39,40,0.8)", width=1), row=1, col=1)
    fig.add_trace(go.Bar(x=lat, y=p * 100, marker_color="#d62728",
                         hovertemplate="%{x:.0f}°: %{y:.2f}% of time<extra></extra>", showlegend=False), row=2, col=1)
    fig.update_xaxes(range=[-180, 180], title="Longitude (°)", dtick=90, row=1, col=1)
    fig.update_yaxes(range=[-90, 90], title="Lat (°)", dtick=45, row=1, col=1)
    fig.update_xaxes(range=[-90, 90], title="Latitude (°)", dtick=30, row=2, col=1)
    fig.update_yaxes(title="% of time", row=2, col=1)
    fig.update_layout(height=470, margin=dict(l=0, r=4, t=10, b=0), bargap=0.05, font=dict(size=11))
    return float_on_dark(lock_for_touch(fig))
