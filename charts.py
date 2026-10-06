"""2D plotly figures for decay, perigee history and the latitude footprint (offline)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import decay
import footprint
from globe import coastline_trace_2d


def decay_figure(h0_km: float, rate_km_day: float, hist=None, height: int = 360):
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
    return fig


def footprint_figure(inclination_deg: float, height: int = 430):
    """World map with the overflown latitude band, next to the share of time spent per latitude."""
    phi = footprint.max_latitude(inclination_deg)
    lat, p = footprint.latitude_distribution(inclination_deg)
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
    return fig


def eol_figure(h0_km: float, rate_km_day: float, bands: list[dict], height: int = 400):
    """Projected altitude of a decaying object against the crowded altitude bands it sinks through."""
    fig = go.Figure()
    proj = decay.altitude_projection(h0_km, rate_km_day, n=400)
    top = h0_km + 40
    colours = {"ISS orbit": "rgba(255,127,14,0.25)"}
    for b in bands:
        if b["lo"] > top:
            continue
        fig.add_hrect(y0=b["lo"], y1=min(b["hi"], top), fillcolor=colours.get(b["label"], "rgba(31,119,180,0.15)"),
                      line_width=0, annotation_text=b["label"], annotation_position="top left",
                      annotation_font_size=11)
    if proj is not None:
        t, h = proj
        fig.add_trace(go.Scatter(x=t, y=h, mode="lines", name="Model projection",
                                 line=dict(color="#d62728", width=3)))
        fig.update_xaxes(title="Days from the element epoch (rough estimate)")
    fig.add_hline(y=decay.REENTRY_ALT_KM, line=dict(color="#888", dash="dash"),
                  annotation_text=f"re-entry threshold {decay.REENTRY_ALT_KM:.0f} km")
    fig.update_yaxes(title="Mean altitude (km)", range=[decay.REENTRY_ALT_KM - 20, top])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig
