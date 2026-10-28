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
BG = "rgba(0,0,0,0)"  # transparent: the page's starfield shows behind the globe
TIER_COLORS = ("#e5484d", "#f5c518", "#30a46c", "#4fd1e0")  # red, yellow, green, cyan (the screened object)


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


GRID_COLOR = "rgba(190, 215, 255, 0.14)"


def float_on_dark(fig: go.Figure) -> go.Figure:
    """Transparent background and light grid lines, so a chart floats on the dark starfield page."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dbe6fb"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig


def ground_track_figure(track, pos, height: int = 460) -> go.Figure:
    """2D ground track on the bundled coastlines (works offline)."""
    fig = go.Figure(coastline_trace_2d())
    fig.add_trace(go.Scatter(x=track["lon"], y=track["lat"], mode="lines", name="3-orbit ground track",
                             line=dict(width=2, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=[pos["lon"]], y=[pos["lat"]], mode="markers", name="Current position",
                             marker=dict(size=11, color="red")))
    # no fixed aspect ratio, so the map fills the full width like the other charts
    fig.update_xaxes(range=[-180, 180], title="Longitude (°)", dtick=60)
    fig.update_yaxes(range=[-90, 90], title="Latitude (°)", dtick=30)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2))
    return float_on_dark(fig)


def _nan_to_null(a):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in a]


_JS = """
(function () {
  const D = __DATA__;
  const gd = document.getElementById('globe');
  const WE = 7.2921159e-5;               // Earth rotation rate, rad/s
  const LINE = D.lineIdx, MARK = D.markIdx;
  const EV = D.ev || null;               // optional conjunction markers
  const DEB = D.deb || null;             // optional debris layer (lines + moving dots), can be switched off
  let showDeb = !!DEB, forceDebLines = true, lastDebLines = -1e9;
  if (DEB) { const tr = gd.data[DEB.lineIdx]; DEB.X = tr.x; DEB.Y = tr.y; DEB.Z = tr.z; }
  let s = 0, playing = D.autoplay !== false, speed = 60, spin = D.spin, last = null, phi = Math.atan2(D.eye[1], D.eye[0]);
  const rad = Math.hypot(D.eye[0], D.eye[1]), ez = D.eye[2];

  // controls
  const bar = document.createElement('div');
  bar.style.cssText = 'position:absolute;top:8px;left:8px;z-index:10;display:flex;gap:8px;align-items:center;' +
    'font:13px sans-serif;color:#dde6f5;background:rgba(10,18,34,.75);padding:6px 10px;border-radius:6px';
  bar.innerHTML = '<button id="pp" style="padding:4px 12px;cursor:pointer">' + (playing ? '⏸ Pause' : '▶ Play') + '</button>' +
    '<label>Speed <select id="sp"><option value="60" selected>60×</option><option value="300">300×</option>' +
    '<option value="1000">1000×</option></select></label>' +
    '<label><input type="checkbox" id="rot"' + (spin ? ' checked' : '') + '> Rotate view</label>' +
    (DEB ? '<label><input type="checkbox" id="deb" checked> Debris orbits (' + DEB.count + ')</label>' : '') +
    '<span id="clk" style="min-width:86px"></span>';
  document.body.appendChild(bar);
  const pp = bar.querySelector('#pp'), sel = bar.querySelector('#sp'), rot = bar.querySelector('#rot'),
        clk = bar.querySelector('#clk');
  pp.onclick = () => { playing = !playing; pp.textContent = playing ? '⏸ Pause' : '▶ Play'; last = null; };
  sel.onchange = () => { speed = +sel.value; };
  rot.onchange = () => { spin = rot.checked; };
  if (DEB) {
    bar.querySelector('#deb').onchange = (e) => {
      showDeb = e.target.checked;
      Plotly.restyle(gd, {visible: showDeb}, [DEB.lineIdx, DEB.markIdx]);
      if (showDeb) { forceDebLines = true; render(0); }
    };
  }
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
    const idx = [LINE, MARK], ux = [lx, mx], uy = [ly, my], uz = [D.Z, mz];
    if (EV) {                            // conjunction points: fixed in space, so they turn with the orbit lines
      const ex = [], ey = [];
      for (let i = 0; i < EV.x.length; i++) { ex.push(EV.x[i] * c + EV.y[i] * sn); ey.push(-EV.x[i] * sn + EV.y[i] * c); }
      idx.push(EV.idx); ux.push(ex); uy.push(ey); uz.push(EV.z);
    }
    if (DEB && showDeb) {                // moving debris dots: every frame
      const dx = [], dy = [], dz = [];
      for (const o of DEB.sats) {
        const f = ((s % o.T) / o.T) * (o.n - 1), i = Math.min(Math.floor(f), o.n - 2), u = f - i, k = o.o + i;
        const x = DEB.X[k] + u * (DEB.X[k + 1] - DEB.X[k]), y = DEB.Y[k] + u * (DEB.Y[k + 1] - DEB.Y[k]);
        dx.push(x * c + y * sn); dy.push(-x * sn + y * c); dz.push(DEB.Z[k] + u * (DEB.Z[k + 1] - DEB.Z[k]));
      }
      idx.push(DEB.markIdx); ux.push(dx); uy.push(dy); uz.push(dz);
    }
    const upd = {x: ux, y: uy, z: uz};
    let lay = {};
    if (spin && dt) {
      phi += dt * 0.105;                  // ~6 degrees/s
      lay = {'scene.camera.eye': {x: rad * Math.cos(phi), y: rad * Math.sin(phi), z: ez}};
    }
    Plotly.update(gd, upd, lay, idx);
    const nowMs = performance.now();
    if (DEB && showDeb && (forceDebLines || nowMs - lastDebLines > 400)) {   // heavy: rotate all debris orbit lines ~2-3x/s
      const n = DEB.X.length, rx = new Array(n), ry = new Array(n);
      for (let i = 0; i < n; i++) {
        const x = DEB.X[i], y = DEB.Y[i];
        if (x === null) { rx[i] = null; ry[i] = null; continue; }
        rx[i] = x * c + y * sn; ry[i] = -x * sn + y * c;
      }
      Plotly.restyle(gd, {x: [rx], y: [ry], z: [DEB.Z]}, [DEB.lineIdx]);
      lastDebLines = nowMs; forceDebLines = false;
    }
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


def _int_or_null(a):
    return [None if not np.isfinite(v) else int(round(float(v))) for v in a]


def orbit_html(paths: list[dict], height: int = 650, highlight: bool = False, spin: bool = True,
               debris: list[dict] | None = None, events: list[dict] | None = None, autoplay: bool = False) -> str:
    """paths: dicts with name, x, y, z (one period, Earth-fixed-at-start km), period_s, optional hover and
    optional `tier` (0 red, 1 yellow, 2 green, 3 cyan; colours every orbit and dot when all paths have one).

    The animation starts paused at time zero (the start time of the paths, e.g. now) at 60x; `autoplay=True` starts it playing.

    `events` marks conjunction points: dicts with x, y, z (km, same axes as the paths), `tier` and `hover`.

    `debris` (same format, many objects) adds a toggleable layer of thin orbit lines and moving dots.
    Returns HTML for st.components.v1.html. The animation loops until paused.
    """
    fig = go.Figure(earth_traces())
    X, Y, Z, line_tier = [], [], [], []
    for p in paths:
        X += _nan_to_null(p["x"]) + [None]
        Y += _nan_to_null(p["y"]) + [None]
        Z += _nan_to_null(p["z"]) + [None]
        line_tier += [p.get("tier", 1)] * (len(p["x"]) + 1)
    tiered = bool(paths) and all("tier" in p for p in paths)  # 0 red, 1 yellow, 2 green
    scale = [[0, TIER_COLORS[0]], [1 / 3, TIER_COLORS[1]], [2 / 3, TIER_COLORS[2]], [1, TIER_COLORS[3]]]
    line_style = (dict(width=2, color=line_tier, colorscale=scale, cmin=0, cmax=3) if tiered else
                  dict(width=5 if highlight else 2, color="#ff5a5f" if highlight else "rgba(255,179,71,0.6)"))
    marker_style = (dict(size=4, color=[p["tier"] for p in paths], colorscale=scale, cmin=0, cmax=3) if tiered else
                    dict(size=7 if highlight else 3.5, color="#ffd23f"))
    fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", hoverinfo="skip", showlegend=False, line=line_style))
    fig.add_trace(go.Scatter3d(
        x=[0] * len(paths), y=[0] * len(paths), z=[0] * len(paths), mode="markers", showlegend=False,
        marker=marker_style, hovertext=[p.get("hover", p["name"]) for p in paths], hoverinfo="text"))
    line_idx, mark_idx = len(fig.data) - 2, len(fig.data) - 1
    ev_data = None
    if events:
        fig.add_trace(go.Scatter3d(
            x=[e["x"] for e in events], y=[e["y"] for e in events], z=[e["z"] for e in events], mode="markers",
            showlegend=False, hovertext=[e["hover"] for e in events], hoverinfo="text",
            marker=dict(size=7, symbol="diamond", color=[e["tier"] for e in events], colorscale=scale, cmin=0, cmax=3,
                        line=dict(width=1, color="#ffffff"))))
        ev_data = {"idx": len(fig.data) - 1, "x": [float(e["x"]) for e in events], "y": [float(e["y"]) for e in events],
                   "z": [float(e["z"]) for e in events]}
    deb_data = None
    if debris:
        DX, DY, DZ, sats = [], [], [], []
        for p in debris:
            n = len(p["x"])
            sats.append({"o": len(DX), "n": n, "T": round(float(p["period_s"]), 1)})
            DX += _int_or_null(p["x"]) + [None]
            DY += _int_or_null(p["y"]) + [None]
            DZ += _int_or_null(p["z"]) + [None]
        fig.add_trace(go.Scatter3d(x=DX, y=DY, z=DZ, mode="lines", hoverinfo="skip", showlegend=False,
                                   opacity=0.06, line=dict(width=1, color="#7f9bd6")))
        fig.add_trace(go.Scatter3d(x=[0] * len(sats), y=[0] * len(sats), z=[0] * len(sats), mode="markers",
                                   hoverinfo="skip", showlegend=False, opacity=0.8, marker=dict(size=2, color="#c3cfe8")))
        deb_data = {"sats": sats, "count": len(sats),  # X/Y/Z are read back from the plotted trace (no second copy)
                    "lineIdx": len(fig.data) - 2, "markIdx": len(fig.data) - 1}
    eye = (1.35, 1.0, 0.6)
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG, uirevision="keep",
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor=BG,
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))))
    data = {
        "X": X, "Y": Y, "Z": Z, "eye": eye, "spin": spin,
        "lineIdx": line_idx, "markIdx": mark_idx, "deb": deb_data, "ev": ev_data, "autoplay": autoplay,
        "sats": [{"x": _nan_to_null(p["x"]), "y": _nan_to_null(p["y"]), "z": _nan_to_null(p["z"]),
                  "T": p["period_s"]} for p in paths],
    }
    js = _JS.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = fig.to_html(full_html=True, include_plotlyjs=True, div_id="globe", post_script=js,
                      config={"displaylogo": False, "scrollZoom": True},
                      default_height=f"{height}px")
    # transparent iframe (no colour-scheme meta: a scheme that differs from the page's makes the browser paint an opaque backdrop)
    return out.replace("<head>", "<head><style>html,body{background:transparent!important}</style>", 1)
