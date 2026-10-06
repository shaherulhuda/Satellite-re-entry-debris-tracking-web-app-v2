"""3D Earth + orbits, rendered as a self-contained HTML page (plotly.js inlined, no network).

The Earth (continents from bundled Natural Earth data) is drawn in the Earth-fixed frame.
A small JavaScript loop moves each object along its orbit and rotates the inertial orbit
lines by the Earth's rotation angle, so the animation runs until the user pauses it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from orbits import EARTH_RADIUS_KM as R

DATA = Path(__file__).with_name("data")
BG = "#05070d"


@lru_cache(maxsize=1)
def _assets():
    mask = np.load(DATA / "land_mask.npy").astype(float) / 255.0
    coast = json.load(open(DATA / "coastlines.json"))
    return mask, coast


def lonlat_to_xyz(lon_deg, lat_deg, r=R):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return (r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat))


def earth_traces() -> list:
    mask, coast = _assets()
    nlat, nlon = mask.shape
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    x, y, z = (np.round(a, 1) for a in lonlat_to_xyz(LON, LAT))
    surface = go.Surface(
        x=x, y=y, z=z, surfacecolor=np.round(mask, 2), cmin=0, cmax=1, showscale=False,
        hoverinfo="skip", colorscale=[[0, "#0a3a6b"], [0.35, "#1b5e9b"], [0.5, "#2f7a3e"], [1, "#6f9a4a"]],
        lighting=dict(ambient=0.75, diffuse=0.55, specular=0.1, roughness=0.9),
    )
    cx, cy, cz = [], [], []
    for ring in coast:
        a = np.array(ring)
        px, py, pz = lonlat_to_xyz(a[:, 0], a[:, 1], R * 1.003)
        cx += list(np.round(px, 1)) + [None]
        cy += list(np.round(py, 1)) + [None]
        cz += list(np.round(pz, 1)) + [None]
    coastlines = go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", hoverinfo="skip", showlegend=False,
                              line=dict(color="rgba(235,245,255,0.8)", width=1))
    return [surface, coastlines]


def coastline_trace_2d() -> go.Scatter:
    _, coast = _assets()
    xs, ys = [], []
    for ring in coast:
        xs += [p[0] for p in ring] + [None]
        ys += [p[1] for p in ring] + [None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#7d8a99", width=1),
                      hoverinfo="skip", showlegend=False)


def ground_track_figure(track, pos, height: int = 520) -> go.Figure:
    """2D ground track on the bundled coastlines (works offline)."""
    fig = go.Figure(coastline_trace_2d())
    fig.add_trace(go.Scatter(x=track["lon"], y=track["lat"], mode="lines", name="3-orbit ground track",
                             line=dict(width=2, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=[pos["lon"]], y=[pos["lat"]], mode="markers", name="Current position",
                             marker=dict(size=11, color="red")))
    fig.update_xaxes(range=[-180, 180], title="Longitude (°)", dtick=30, constrain="domain")
    fig.update_yaxes(range=[-90, 90], title="Latitude (°)", dtick=30, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.15))
    return fig


def _nan_to_null(a):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in a]


_JS = """
(function () {
  const D = __DATA__;
  const gd = document.getElementById('globe');
  const WE = 7.2921159e-5;               // Earth rotation rate, rad/s
  const LINE = D.lineIdx, MARK = D.markIdx;
  let s = 0, playing = true, speed = 300, spin = D.spin, last = null, phi = Math.atan2(D.eye[1], D.eye[0]);
  const rad = Math.hypot(D.eye[0], D.eye[1]), ez = D.eye[2];

  // controls
  const bar = document.createElement('div');
  bar.style.cssText = 'position:absolute;top:8px;left:8px;z-index:10;display:flex;gap:8px;align-items:center;' +
    'font:13px sans-serif;color:#dde6f5;background:rgba(10,18,34,.75);padding:6px 10px;border-radius:6px';
  bar.innerHTML = '<button id="pp" style="padding:4px 12px;cursor:pointer">⏸ Pause</button>' +
    '<label>Speed <select id="sp"><option value="60">60×</option><option value="300" selected>300×</option>' +
    '<option value="1000">1000×</option></select></label>' +
    '<label><input type="checkbox" id="rot"' + (spin ? ' checked' : '') + '> Rotate view</label>' +
    '<span id="clk" style="min-width:86px"></span>';
  document.body.appendChild(bar);
  const pp = bar.querySelector('#pp'), sel = bar.querySelector('#sp'), rot = bar.querySelector('#rot'),
        clk = bar.querySelector('#clk');
  pp.onclick = () => { playing = !playing; pp.textContent = playing ? '⏸ Pause' : '▶ Play'; last = null; };
  sel.onchange = () => { speed = +sel.value; };
  rot.onchange = () => { spin = rot.checked; };
  gd.addEventListener('pointerdown', () => { spin = false; rot.checked = false; });

  function render(dt) {
    const th = WE * s, c = Math.cos(th), sn = Math.sin(th);
    const lx = new Array(D.X.length), ly = new Array(D.X.length);
    for (let i = 0; i < D.X.length; i++) {
      const x = D.X[i], y = D.Y[i];
      if (x === null) { lx[i] = null; ly[i] = null; continue; }
      lx[i] = x * c + y * sn; ly[i] = -x * sn + y * c;
    }
    const mx = [], my = [], mz = [];
    for (const o of D.sats) {
      const n = o.x.length - 1, f = ((s % o.T) / o.T) * n, i = Math.min(Math.floor(f), n - 1), u = f - i;
      const x = o.x[i] + u * (o.x[i + 1] - o.x[i]), y = o.y[i] + u * (o.y[i + 1] - o.y[i]);
      mx.push(x * c + y * sn); my.push(-x * sn + y * c); mz.push(o.z[i] + u * (o.z[i + 1] - o.z[i]));
    }
    const upd = {x: [lx, mx], y: [ly, my], z: [D.Z, mz]};
    let lay = {};
    if (spin && dt) {
      phi += dt * 0.105;                  // ~6 degrees/s
      lay = {'scene.camera.eye': {x: rad * Math.cos(phi), y: rad * Math.sin(phi), z: ez}};
    }
    Plotly.update(gd, upd, lay, [LINE, MARK]);
    const m = Math.floor(s / 60);
    clk.textContent = '+' + String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
  }

  function tick(now) {
    requestAnimationFrame(tick);
    if (!playing) return;
    if (last === null) last = now;
    const dt = Math.min((now - last) / 1000, 0.25);
    last = now;
    s += dt * speed;
    render(dt);
  }
  render(0);
  requestAnimationFrame(tick);
})();
"""


def orbit_html(paths: list[dict], height: int = 650, highlight: bool = False, spin: bool = True) -> str:
    """paths: dicts with name, x, y, z (one period, Earth-fixed-at-start km), period_s, optional hover.

    Returns HTML for st.components.v1.html. The animation loops until paused.
    """
    fig = go.Figure(earth_traces())
    X, Y, Z = [], [], []
    for p in paths:
        X += _nan_to_null(p["x"]) + [None]
        Y += _nan_to_null(p["y"]) + [None]
        Z += _nan_to_null(p["z"]) + [None]
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(width=5 if highlight else 2, color="#ff5a5f" if highlight else "rgba(255,179,71,0.6)")))
    fig.add_trace(go.Scatter3d(
        x=[0] * len(paths), y=[0] * len(paths), z=[0] * len(paths), mode="markers", showlegend=False,
        marker=dict(size=7 if highlight else 3.5, color="#ffd23f"),
        hovertext=[p.get("hover", p["name"]) for p in paths], hoverinfo="text"))
    eye = (1.35, 1.0, 0.6)
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG, uirevision="keep",
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor=BG,
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))))
    data = {
        "X": X, "Y": Y, "Z": Z, "eye": eye, "spin": spin,
        "lineIdx": len(fig.data) - 2, "markIdx": len(fig.data) - 1,
        "sats": [{"x": _nan_to_null(p["x"]), "y": _nan_to_null(p["y"]), "z": _nan_to_null(p["z"]),
                  "T": p["period_s"]} for p in paths],
    }
    js = _JS.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    return fig.to_html(full_html=True, include_plotlyjs=True, div_id="globe", post_script=js,
                       config={"displaylogo": False, "scrollZoom": True},
                       default_height=f"{height}px")
