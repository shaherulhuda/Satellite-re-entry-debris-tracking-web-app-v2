"""Re-Entry & Debris Mitigation Assistant (Streamlit UI). Physics lives in the other modules."""
import base64
import html
import os
import urllib.parse
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"


def _data_uri(name: str) -> str:
    mime = "image/svg+xml" if name.endswith(".svg") else ("image/jpeg" if name.endswith(".jpg") else "image/png")
    return f"data:{mime};base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


st.set_page_config(page_title="Orbisk: Re-Entry & Debris Mitigation", page_icon=str(ASSETS / "orbisk-icon.png"), layout="wide")

entered = bool(st.session_state.get("entered", False))

_CSS = """<style>
/* ---- space look: starfield page, glass panels, pill tabs, matching scrollbars, tooltips and motion ---- */
.stApp { background: #070b1a url(__BG__) center / cover fixed no-repeat; }
header.stAppHeader, header.stAppHeader * { background: transparent !important; pointer-events: none !important; }  /* the invisible bar must not block clicks under it (the assistant button) */
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] * { pointer-events: auto !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stDecoration"],
footer, [data-testid="stFooter"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }
iframe { border-radius: 14px; }

/* glass panels: soft blur, thin glowing border */
.st-key-summary_card, .st-key-pick_actions, [class*="st-key-pass_card_"], [data-testid="stExpander"], [data-testid="stPopoverBody"], [data-testid="stForm"] {
    background: rgba(12, 20, 44, .55) !important; backdrop-filter: blur(10px) saturate(120%); -webkit-backdrop-filter: blur(10px) saturate(120%);
    border: 1px solid rgba(120, 200, 255, .20) !important; border-radius: 14px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, .35), inset 0 0 22px rgba(79, 209, 224, .05); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(120, 200, 255, .20); box-shadow: 0 8px 30px rgba(0, 0, 0, .35); }

/* tabs: pills, orange glow on the active one (sub-tabs are smaller) */
[role="tablist"] { gap: .4rem; border-bottom: 0 !important; box-shadow: none !important; }
[data-testid="stTab"] { border-radius: 999px; padding: .4rem 1.05rem; background: rgba(255, 255, 255, .04);
    border: 1px solid rgba(255, 255, 255, .08); transition: background .2s, box-shadow .2s, border-color .2s; }
[data-testid="stTab"]:hover { background: rgba(79, 209, 224, .12); border-color: rgba(79, 209, 224, .35); }
[data-testid="stTab"][data-selected="true"] { background: rgba(255, 122, 61, .14); border-color: rgba(255, 122, 61, .6);
    box-shadow: 0 0 18px rgba(255, 122, 61, .35); }
[data-testid="stTab"][data-selected="true"] p { color: #fff; font-weight: 600; }
.react-aria-SelectionIndicator { background: #FF7A3D !important; height: 2px !important; border-radius: 2px;
    box-shadow: 0 0 10px 1px rgba(255, 122, 61, .9); }
[role="tabpanel"] [role="tablist"] [data-testid="stTab"] { padding: .25rem .8rem; }
[role="tabpanel"] [role="tablist"] [data-testid="stTab"] p { font-size: .85rem; }

/* section switcher (segmented control) drawn as pills, like the tabs: orange glow on the selected one; the tracker's view switch is smaller */
[data-testid="stButtonGroup"] [role="radiogroup"] { gap: .4rem; background: transparent !important; border: 0 !important; padding: 0 !important; }
[data-testid="stButtonGroup"] button[role="radio"] { border-radius: 999px !important; padding: .4rem 1.05rem; background: rgba(255, 255, 255, .04) !important;
    border: 1px solid rgba(255, 255, 255, .08) !important; transition: background .2s, box-shadow .2s, border-color .2s; }
[data-testid="stButtonGroup"] button[role="radio"]:hover { background: rgba(79, 209, 224, .12) !important; border-color: rgba(79, 209, 224, .35) !important; }
[data-testid="stButtonGroup"] button[role="radio"][data-selected="true"] { background: rgba(255, 122, 61, .14) !important;
    border-color: rgba(255, 122, 61, .6) !important; box-shadow: 0 0 18px rgba(255, 122, 61, .35), 0 2px 0 #FF7A3D; }
[data-testid="stButtonGroup"] button[role="radio"][data-selected="true"] p { color: #fff; font-weight: 600; }
.st-key-tracker_view button[role="radio"] { padding: .25rem .8rem; }
.st-key-tracker_view button[role="radio"] p { font-size: .85rem; }
.st-key-section { margin-bottom: .4rem; }

/* scrollbars and tooltips */
* { scrollbar-width: thin; scrollbar-color: rgba(79, 209, 224, .5) transparent; }
::-webkit-scrollbar { width: 10px; height: 10px; background: transparent; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, rgba(79, 209, 224, .6), rgba(255, 122, 61, .5)); border-radius: 8px;
    border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-corner { background: transparent; }
[data-testid="stTooltipContent"] { background: rgba(10, 16, 36, .96) !important; color: #e8eefc !important;
    border: 1px solid rgba(79, 209, 224, .4); border-radius: 8px; box-shadow: 0 4px 18px rgba(0, 0, 0, .5); }

[data-testid="stSkeleton"] { background: rgba(120, 200, 255, .08) !important; border-radius: 12px !important; }

/* motion: cards rise in, pass cards lift on hover, high-risk badge glows */
@keyframes orbisk-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.st-key-summary_card, .st-key-pick_actions, [class*="st-key-pass_card_"] { animation: orbisk-rise .5s ease backwards; transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
[class*="st-key-pass_card_"]:hover { transform: translateY(-4px); border-top-color: rgba(79, 209, 224, .55) !important;
    border-right-color: rgba(79, 209, 224, .55) !important; border-bottom-color: rgba(79, 209, 224, .55) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, .45), 0 0 22px rgba(79, 209, 224, .25), inset 0 0 22px rgba(79, 209, 224, .08); }
@keyframes orbisk-glow { 0%, 100% { box-shadow: 0 0 6px 0 rgba(229, 72, 77, .55); } 50% { box-shadow: 0 0 18px 3px rgba(229, 72, 77, .85); } }
.st-key-risk_badge_red .stMarkdownBadge { animation: orbisk-glow 2.2s ease-in-out infinite; }
.st-key-risk_badge_yellow .stMarkdownBadge { box-shadow: 0 0 10px rgba(224, 168, 0, .45); }
.st-key-risk_badge_green .stMarkdownBadge { box-shadow: 0 0 10px rgba(48, 164, 108, .4); }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* header: logo top-left with the tagline beside it */
.orbisk-header { display: flex; align-items: center; gap: 1rem; margin: 0 0 .75rem; }
.orbisk-header img { height: 52px; width: auto; }
.orbisk-header .tagline { font-size: 1.05rem; line-height: 1.25; opacity: .8; border-left: 1px solid rgba(160, 200, 255, .35); padding-left: 1rem; }
@media (max-width: 640px) { .orbisk-header .tagline { display: none; } }

/* floating assistant button: planet icon with a white label under it; hidden label while the popup is open */
.st-key-ai_fab { position: fixed; top: 1rem; right: 1.5rem; z-index: 1000001; width: auto !important; touch-action: none; cursor: grab; }
.st-key-ai_fab button { width: 3.5rem; height: 3.5rem; padding: 0; border: 0; border-radius: 50%; font-size: 0;
    background: url(__ICON__) center / contain no-repeat; }
.st-key-ai_fab button:hover, .st-key-ai_fab button:focus, .st-key-ai_fab button:active {
    background-color: transparent !important; border: 0 !important; box-shadow: none !important; }
.st-key-ai_fab button > * { display: none; }
.st-key-ai_fab::after { content: "Ask Orbisk AI"; position: absolute; top: calc(100% + 2px); left: 50%; transform: translateX(-50%);
    white-space: nowrap; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; line-height: 1.3;
    pointer-events: none; background: #ffffff; color: #0a0e1f; box-shadow: 0 1px 6px rgba(0,0,0,.35); }
.st-key-ai_fab:has(button[aria-expanded="true"])::after { display: none; }
[data-testid="stPopoverBody"] { width: min(520px, 92vw); max-height: calc(100vh - 8rem); overflow-y: auto; }

/* tables are read-only: hide the cell pop-up box that would otherwise linger while a clicked row opens the tracker */
[class*="gdg-"]:has(> .gdg-clip-region) { display: none !important; }

/* one size scale: headings 1.35rem, numbers 1.5rem, labels and card detail 0.85-0.9rem */
h3 { font-size: 1.35rem !important; }
h4 { font-size: 1.2rem !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] p { font-size: .9rem !important; }

/* loading indicator: a small planet with a moving orbit (kept visible, bottom-left, now that the top bar is gone) */
[data-testid="stStatusWidget"] { position: fixed !important; left: 1rem; bottom: 1rem; top: auto !important; right: auto !important; z-index: 1000;
    background: rgba(12, 20, 44, .7); border: 1px solid rgba(120, 200, 255, .25); border-radius: 999px; padding: .25rem .6rem; backdrop-filter: blur(8px); }
[data-testid="stStatusWidgetRunningIcon"] { position: relative; width: 1.6rem; height: 1.6rem; }
[data-testid="stStatusWidgetRunningIcon"] svg { display: none; }
[data-testid="stStatusWidgetRunningIcon"]::before { content: ""; position: absolute; inset: 0.5rem; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #5FE0C8, #14456B); }
[data-testid="stStatusWidgetRunningIcon"]::after { content: ""; position: absolute; inset: 0; border-radius: 50%;
    border: 1.5px solid rgba(79, 209, 224, .55); box-sizing: border-box; animation: orbisk-orbit 1.2s linear infinite;
    background: radial-gradient(circle at 50% 0, #FF7A3D 0 2.6px, transparent 3.2px); }
@keyframes orbisk-orbit { to { transform: rotate(360deg); } }

/* cover page: shown first, until "Enter Watchlist" is pressed */
.orbisk-cover { text-align: center; max-width: 980px; margin: 0 auto; padding-top: 1.2rem; }
.orbisk-cover .hero { position: relative; height: 250px; margin-bottom: .6rem; }
.orbisk-cover .orbit { position: absolute; left: 50%; top: 50%; border-radius: 50%; transform: translate(-50%, -50%); }
.orbisk-cover .planet { width: 118px; height: 118px; background: radial-gradient(circle at 35% 30%, #5FE0C8, #3FA796 45%, #14456B);
    box-shadow: 0 0 50px rgba(79, 209, 224, .35); animation: orbisk-breathe 6s ease-in-out infinite; }
.orbisk-cover .ring { border: 1.5px solid rgba(79, 209, 224, .4); animation: orbisk-orbit 16s linear infinite; }
.orbisk-cover .ring.r1 { width: 190px; height: 190px; background: radial-gradient(circle at 50% 0, #FF7A3D 0 5px, transparent 6px); animation-duration: 9s; }
.orbisk-cover .ring.r2 { width: 250px; height: 250px; border-color: rgba(255, 122, 61, .28); animation-direction: reverse;
    background: radial-gradient(circle at 50% 0, #4FD1E0 0 4px, transparent 5px); }
.orbisk-cover .ring.r3 { width: 320px; height: 320px; border-style: dashed; border-color: rgba(190, 215, 255, .16); animation-duration: 28s;
    background: radial-gradient(circle at 50% 0, #ffffff 0 3px, transparent 4px); }
.orbisk-cover img.logo { height: 58px; width: auto; margin-bottom: 1rem; }
.orbisk-cover h1 { font-size: clamp(1.7rem, 4.2vw, 2.7rem); line-height: 1.2; margin: .2rem 0 .9rem; font-weight: 700; letter-spacing: -.01em;
    text-shadow: 0 0 30px rgba(79, 209, 224, .25); }
.orbisk-cover .lead { font-size: 1.1rem; line-height: 1.55; color: rgba(232, 238, 252, .82); max-width: 740px; margin: 0 auto 1.6rem; }
.orbisk-cover .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: left; margin-bottom: 1.8rem; }
.orbisk-cover .ccard { background: rgba(12, 20, 44, .55); backdrop-filter: blur(10px); border: 1px solid rgba(120, 200, 255, .2); border-radius: 14px;
    padding: 1rem 1.1rem; box-shadow: 0 8px 30px rgba(0, 0, 0, .35), inset 0 0 22px rgba(79, 209, 224, .05); transition: transform .2s ease, border-color .2s ease; }
.orbisk-cover .ccard:hover { transform: translateY(-4px); border-color: rgba(79, 209, 224, .55); }
.orbisk-cover .ccard b { display: block; font-size: 1.02rem; margin-bottom: .3rem; }
.orbisk-cover .ccard p { margin: 0; font-size: .9rem; line-height: 1.45; color: rgba(232, 238, 252, .75); }
.orbisk-cover .fine { font-size: .8rem; color: rgba(232, 238, 252, .5); margin-top: 1.2rem; }
.orbisk-cover > * { animation: orbisk-rise .7s ease backwards; }
.orbisk-cover > :nth-child(2) { animation-delay: .1s; } .orbisk-cover > :nth-child(3) { animation-delay: .2s; }
.orbisk-cover > :nth-child(4) { animation-delay: .3s; } .orbisk-cover > :nth-child(5) { animation-delay: .4s; }
.st-key-enter_btn { max-width: 460px; margin: 0 auto; }
.st-key-enter_btn button { font-size: 1.3rem; font-weight: 700; padding: .95rem 1.5rem; border-radius: 999px; letter-spacing: .01em;
    box-shadow: 0 0 30px rgba(43, 181, 200, .55); animation: orbisk-cta 2.6s ease-in-out infinite; }
.st-key-enter_btn button p { font-size: 1.3rem; font-weight: 700; }
.st-key-enter_btn button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 0 44px rgba(79, 209, 224, .8); }
@keyframes orbisk-orbit { to { transform: translate(-50%, -50%) rotate(360deg); } }
@keyframes orbisk-breathe { 0%, 100% { transform: translate(-50%, -50%) scale(1); } 50% { transform: translate(-50%, -50%) scale(1.06); } }
@keyframes orbisk-cta { 0%, 100% { box-shadow: 0 0 22px rgba(43, 181, 200, .45); } 50% { box-shadow: 0 0 42px rgba(43, 181, 200, .85); } }
@media (max-width: 760px) { .orbisk-cover .cards { grid-template-columns: 1fr; } .orbisk-cover .hero { height: 200px; } }
</style>""".replace("__BG__", _data_uri("orbisk-bg.jpg")).replace("__ICON__", _data_uri("orbisk-mark.png"))
# one unindented line of HTML: indented or multi-line HTML is rendered by markdown as a code block
st.markdown(
    _CSS
    + (
        '<div class="orbisk-header">'
        + f'<img alt="Orbisk" src="{_data_uri("orbisk-logo-for-dark.png")}">'
        + '<div class="tagline">Re-entry &amp; debris<br>mitigation assistant</div></div>'
        if entered else ""
    ),
    unsafe_allow_html=True,
)



def _enter():
    st.session_state["entered"] = True
    st.session_state["section"] = "🛰️ Watchlist"


if not entered:
    # the cover: one-line HTML pieces (indented or multi-line HTML would be shown as code)
    st.markdown(
        '<div class="orbisk-cover">'
        '<div class="hero"><i class="orbit ring r3"></i><i class="orbit ring r2"></i><i class="orbit ring r1"></i><i class="orbit planet"></i></div>'
        f'<div><img class="logo" alt="Orbisk" src="{_data_uri("orbisk-logo-for-dark.png")}"></div>'
        '<div><h1>Which satellites are coming down, and what could they hit?</h1>'
        '<p class="lead">Orbisk ranks the 50 highest-risk active satellites in low Earth orbit, estimates when each might re-enter the atmosphere, '
        'and screens them for close approaches with other satellites and space debris.</p></div>'
        '<div class="cards">'
        '<div class="ccard"><b>🛰️ Watchlist</b><p>The 50 highest-risk satellites, ranked and colour-coded: 10 red, 30 yellow, 10 green. Pick one to dig in.</p></div>'
        '<div class="ccard"><b>📉 Re-entry Tracker</b><p>See an orbit in 3D, how fast it is decaying, and the latitudes it can come down over.</p></div>'
        '<div class="ccard"><b>⚠️ Conjunctions &amp; Triage</b><p>Screen one satellite against every other satellite and debris fragment for close passes in the next 1 to 30 days.</p></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    _l, _m, _r = st.columns([1, 1.6, 1])
    _m.button("Enter Watchlist  →", key="enter_btn", type="primary", on_click=_enter, width="stretch")
    st.markdown(
        '<div class="orbisk-cover"><p class="fine">Built on public CelesTrak element sets. Estimates and screening are triage aids, not operational decisions.</p></div>',
        unsafe_allow_html=True,
    )

# Loading screen: shown first (before the heavy imports and data load) on the first run of a session, replacing Streamlit's
# grey placeholder bars. It removes itself when the data is ready, and fades away by itself after 25 s so an error can never stay hidden.
boot = st.empty()
if entered and "_booted" not in st.session_state:
    boot.markdown(
        """<style>
        .orbisk-boot { position: fixed; inset: 0; z-index: 100000; display: flex; flex-direction: column; align-items: center; justify-content: center;
            gap: 1.6rem; background: radial-gradient(ellipse at 70% 60%, #12224a 0%, #070b1a 62%); animation: orbisk-boot-out .6s ease 25s forwards; }
        .orbisk-boot img { height: 64px; width: auto; }
        .orbisk-boot .orbit { position: relative; width: 96px; height: 96px; }
        .orbisk-boot .orbit i { position: absolute; border-radius: 50%; }
        .orbisk-boot .planet { inset: 30px; background: radial-gradient(circle at 35% 30%, #5FE0C8, #14456B); animation: orbisk-beat 1.8s ease-in-out infinite; }
        .orbisk-boot .ring { inset: 0; border: 1.5px solid rgba(79, 209, 224, .55); animation: orbisk-spin 1.6s linear infinite;
            background: radial-gradient(circle at 50% 0, #FF7A3D 0 4px, transparent 5px); }
        .orbisk-boot .ripple { inset: 0; border: 1px solid rgba(79, 209, 224, .5); animation: orbisk-ripple 1.8s ease-out infinite; }
        .orbisk-boot p { margin: 0; font-size: .9rem; letter-spacing: .08em; color: rgba(232, 238, 252, .7); }
        @keyframes orbisk-spin { to { transform: rotate(360deg); } }
        @keyframes orbisk-beat { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.18); } }
        @keyframes orbisk-ripple { from { transform: scale(.6); opacity: .9; } to { transform: scale(1.5); opacity: 0; } }
        @keyframes orbisk-boot-out { to { opacity: 0; visibility: hidden; pointer-events: none; } }
        </style>"""
        + '<div class="orbisk-boot">'
        + f'<img alt="Orbisk" src="{_data_uri("orbisk-logo-for-dark.png")}">'
        + '<div class="orbit"><i class="ripple"></i><i class="planet"></i><i class="ring"></i></div>'
        + "<p>LOADING ORBITAL DATA</p></div>",
        unsafe_allow_html=True,
    )

import numpy as np  # noqa: E402  (imported after the loading screen so it appears immediately)
import pandas as pd  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from sgp4.api import jday  # noqa: E402

import charts  # noqa: E402
import conjunction  # noqa: E402
import dataset  # noqa: E402
import decay  # noqa: E402
import footprint  # noqa: E402
import globe  # noqa: E402
import history  # noqa: E402
import llm  # noqa: E402
import orbits  # noqa: E402
import triage  # noqa: E402

WATCHLIST_SIZE = 50  # the watchlist always shows the 50 highest-risk active satellites


@st.cache_data
def get_data():
    """Active satellites + every debris group, with the stored closest-approach results attached."""
    cat, recs = dataset.load_catalogue()
    ca = dataset.load_closest_approaches()
    meta = {k: ca[k] for k in ("start_utc", "window_hours", "threshold_km", "top")} if ca else None
    return dataset.attach_closest_approaches(cat, ca), recs, meta


@st.cache_data
def get_history():
    return history.load_history()


df, records, ca_meta = get_data()
hist_all = get_history()
components_df = decay.risk_components(df)
boot.empty()  # data is ready: remove the loading screen
st.session_state["_booted"] = True
if not entered:
    st.stop()  # the cover has been shown while the data warmed up; nothing else is built until "Enter Watchlist"

SEC_WATCH, SEC_TRACK, SEC_TRIAGE = "🛰️ Watchlist", "📉 Re-entry Tracker", "⚠️ Conjunctions & Triage"
VIEW_ORBIT, VIEW_DECAY, VIEW_FOOT = "Orbit", "Decay", "Re-entry footprint"


@st.cache_data(ttl=300)
def watchlist_paths(norad_ids: tuple, tiers: tuple = (), risks: tuple = ()):
    out = []
    for k, nid in enumerate(norad_ids):
        rec = records[nid]
        try:
            p = orbits.orbit_path_xyz(rec, n_points=120)
        except Exception:
            continue
        row = df[df["NORAD_CAT_ID"] == nid].iloc[0]
        p["name"] = rec["OBJECT_NAME"]
        p["hover"] = f"{rec['OBJECT_NAME']}<br>perigee {row['perigee_km']:.0f} km · apogee {row['apogee_km']:.0f} km"
        if tiers:
            p["tier"] = int(tiers[k])
            p["hover"] += f"<br>risk {risks[k]:.0f}"
        out.append(p)
    return out


@st.cache_data(ttl=300, max_entries=12)
def watchlist_globe_html(norad_ids: tuple, tiers: tuple, risks: tuple, debris_ids: tuple) -> str:
    """The finished watchlist globe: rebuilt only when the objects shown change."""
    return globe.orbit_html(watchlist_paths(norad_ids, tiers, risks), debris=debris_orbit_paths(debris_ids))


def now_key() -> str:
    """The current UTC minute. The tracker's globe, position and ground track all use it, so they agree."""
    return pd.Timestamp.now("UTC").floor("min").strftime("%Y-%m-%d %H:%M")


def _minute(key: str):
    return orbits.TS.utc(*pd.Timestamp(key).timetuple()[:5])


@st.cache_data(ttl=300, max_entries=24)
def tracker_geometry(norad: int, when: str):
    """(position, 3-orbit ground track) at `when`, or the error text if the object cannot be propagated."""
    try:
        t0 = _minute(when)
        return orbits.current_position(records[int(norad)], when=t0), orbits.ground_track(records[int(norad)], n_orbits=3, start=t0), None
    except Exception as exc:  # e.g. decayed / bad elements
        return None, None, str(exc)


@st.cache_data(ttl=300, max_entries=12)
def tracker_globe_html(norad: int, when: str) -> str:
    """The tracker globe for one object, starting paused at `when`; rebuilt only for a new object or minute."""
    rec = records[int(norad)]
    path = orbits.orbit_path_xyz(rec, n_points=360, start=_minute(when))
    path["name"] = rec["OBJECT_NAME"]
    return globe.orbit_html([path], height=600, highlight=True)


@st.cache_data(ttl=300)
def debris_orbit_paths(debris_ids: tuple):
    """Orbit lines for the given debris fragments (empty tuple = none)."""
    return orbits.orbit_paths_fast([records[int(i)] for i in debris_ids], n_points=48) if debris_ids else []


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
    pasted = (st.session_state.get("pasted_key") or "").strip()
    configured = (_secret("ANTHROPIC_API_KEY") or "").strip()
    if not (pasted or configured):
        return False, None, "AI is off: add an API key in the assistant (top right) to enable this."
    if configured and not pasted:
        gate = _secret("ASSISTANT_PASSWORD")
        if gate and st.session_state.get("ai_gate") != gate:
            return False, None, "Enter the access code in the assistant (top right) to enable AI features."
    return True, pasted or configured or None, None


ASSISTANT_DEFAULT_MAX = 10  # questions per session (one question, briefing or explanation = one request)


def ai_guarded(fn):
    """Run one model call with the shared per-session request cap and friendly errors. None on failure."""
    ready, key, msg = ai_status()
    if not ready:
        st.info(msg)
        return None
    max_req = int(_secret("ASSISTANT_MAX_REQUESTS") or ASSISTANT_DEFAULT_MAX)
    if st.session_state.setdefault("ai_requests", 0) >= max_req:
        st.error(f"You have used all {max_req} assistant questions for this session.")
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


def sticky(widget, key: str, *args, **kwargs):
    """Create a widget whose value survives while its section is off screen (Streamlit forgets widgets that are not drawn)."""
    mem = st.session_state.setdefault("_mem", {})
    if key not in st.session_state and key in mem:
        st.session_state[key] = mem[key]
        for stale in ("value", "index", "default"):  # the remembered value wins
            kwargs.pop(stale, None)
    value = widget(*args, key=key, **kwargs)
    mem[key] = value
    return value


def go_to(section: str, **picks):
    """Open another section with a few widget values preset, then rerun the whole app. Call from a button handler."""
    st.session_state["_goto"] = section
    st.session_state.setdefault("_picks", {})[section] = picks
    st.rerun()


# data shared by every section
weights = decay.DEFAULT_WEIGHTS  # fixed: not adjustable in the app
scored = df.copy()
scored["risk"] = decay.risk_score(components_df, weights)
sats = scored[(scored["source"] == "active") & scored["leo"]]  # active LEO satellites only (not debris, not GEO/MEO)
# the same 50 highest-risk active satellites everywhere (watchlist pool, tracker and screening pickers), highest risk first
ids = sats.sort_values("risk", ascending=False).head(WATCHLIST_SIZE)["NORAD_CAT_ID"].astype(int).tolist()
default = 0

# one section at a time: only the section on screen is built (a tab bar would build all three on every click)
if "_goto" in st.session_state:
    st.session_state["section"] = st.session_state.pop("_goto")
section = st.segmented_control("Section", [SEC_WATCH, SEC_TRACK, SEC_TRIAGE], key="section", default=SEC_WATCH, required=True,
                               label_visibility="collapsed")


# ---------------------------------------------------------------- Watchlist
@st.fragment
def watchlist_view():
    st.subheader("Watchlist")
    c_mode, c_n = st.columns([3, 1])
    mode = sticky(c_mode.selectbox, "w_mode", "Rank by", ["Risk score: highest first", "Risk score: lowest first",
                                                          "Soonest estimated re-entry"],
                  help="The watchlist draws on the 50 highest-risk active satellites; this orders them and the first "
                       "'Objects' of that order are shown.")
    n_show = sticky(c_n.number_input, "w_n", "Objects", min_value=10, max_value=WATCHLIST_SIZE, value=10, step=10,
                    help="How many to show: the first N of the 50 highest-risk satellites in the chosen order.")
    n_deb_total = int((df["source"] == "debris").sum())
    show_debris = sticky(st.toggle, "w_debris", "Show debris orbits on globe", value=False,
                         help="Debris is not part of the watchlist; this only draws debris orbits on the globe for context.")
    n_deb_show = 0
    if show_debris:
        n_deb_show = sticky(st.slider, "w_deb_n", "Debris orbits to draw", min_value=50, max_value=n_deb_total,
                            value=min(50, n_deb_total), step=50, help="Highest-risk debris first.")

    def sort_by_mode(frame: pd.DataFrame) -> pd.DataFrame:
        if mode.startswith("Risk"):
            return frame.sort_values("risk", ascending=mode.endswith("lowest first"))
        return frame.sort_values("decay_days", na_position="last")  # soonest estimated re-entry

    # the pool is always the WATCHLIST_SIZE highest-risk satellites; the order option sorts them and Objects shows the first N
    top = sort_by_mode(sats.sort_values("risk", ascending=False).head(WATCHLIST_SIZE)).head(int(n_show))

    levels = decay.traffic_light(sats["risk"]).loc[top.index]  # level = risk rank among the satellites, whatever is listed
    LEVEL_LABEL = {"red": "🔴 High", "yellow": "🟡 Medium", "green": "🟢 Low"}
    hours_txt = ["—" if pd.isna(h) else f"{h:.1f} h" for h in top["ca_hours"]]
    table = pd.DataFrame({
        "Level": levels.map(LEVEL_LABEL),
        "Name": top["OBJECT_NAME"],
        "Perigee (km)": top["perigee_km"], "Apogee (km)": top["apogee_km"], "Inclination (°)": top["INCLINATION"],
        "Est. time to re-entry": top["decay_days"].map(decay.format_days),
        "Range (±×2)": [decay.format_range(lo, hi) for lo, hi in zip(top["decay_days_low"], top["decay_days_high"])],
        "Closest approach (km)": top["ca_miss_km"], "Rel. speed (km/s)": top["ca_speed_km_s"],
        "Time to closest approach": hours_txt,
        "Risk score": top["risk"],
    })
    table_ids = tuple(int(i) for i in top["NORAD_CAT_ID"])
    debris_ranked = scored[scored["source"] == "debris"].sort_values("risk", ascending=False)
    deb_ids = tuple(int(i) for i in debris_ranked["NORAD_CAT_ID"])[:int(n_deb_show)]  # debris only appears as globe orbits
    tier_idx = tuple({"red": 0, "yellow": 1, "green": 2}[l] for l in levels)
    bar_hex = {"red": "#e5484d", "yellow": "#e0a800", "green": "#30a46c"}

    def risk_bar(score: float, level: str) -> str:
        """Progress-bar image (data URI): track + fill in the level colour + the score. The leading comment holds the
        zero-padded score so sorting the column orders rows by score."""
        fill = 150 * score / 100
        svg = (f'<!--{round(score * 10):04d}--><svg xmlns="http://www.w3.org/2000/svg" width="210" height="28" viewBox="0 0 210 28">'
               f'<rect x="0" y="11" width="150" height="6" rx="3" fill="#808080" fill-opacity="0.28"/>'
               f'<rect x="0" y="11" width="{fill:.1f}" height="6" rx="3" fill="{bar_hex[level]}"/>'
               f'<text x="162" y="18" font-family="Source Sans Pro, Arial, sans-serif" font-size="12" font-weight="600" '
               f'fill="{bar_hex[level]}">{score:.0f}</text></svg>')
        return "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe="")

    shown_table = table.assign(**{"Risk score": [risk_bar(s, l) for s, l in zip(top["risk"], levels)]})
    event = st.dataframe(
        shown_table, width="stretch", hide_index=True, key=f"watch_table_{st.session_state.get('_watch_gen', 0)}", on_select="rerun",
        selection_mode=["single-row", "single-cell"],  # click any cell (or tick the row) to open the object
        column_config={
            "Perigee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Apogee (km)": st.column_config.NumberColumn(format="%.1f"),
            "Inclination (°)": st.column_config.NumberColumn(format="%.2f"),
            "Closest approach (km)": st.column_config.NumberColumn(format="%.2f"),
            "Rel. speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
            "Risk score": st.column_config.ImageColumn("Risk score", width="medium"),
        },
    )
    # clicking a row (any cell, or its tick box) offers two next steps for that satellite
    picked = list(event.selection.rows) or [c[0] if isinstance(c, (tuple, list)) else c.get("row") for c in event.selection.cells]
    picked_id = table_ids[picked[0]] if picked and picked[0] < len(table_ids) else None
    if picked_id is not None:
        with st.container(border=True, key="pick_actions"):
            p1, p2, p3 = st.columns([2, 1.3, 1.5], vertical_alignment="center")
            p1.markdown(f"**{records[picked_id]['OBJECT_NAME']}**  \n<span style=\"font-size:.85rem;opacity:.7\">What next?</span>",
                        unsafe_allow_html=True)
            go_foot = p2.button("Ground re-entry footprint", key="pick_footprint", type="primary", width="stretch")
            go_screen = p3.button("Screen for close approaches", key="pick_screen", type="primary", width="stretch")
        if go_foot or go_screen:
            st.session_state["_watch_gen"] = st.session_state.get("_watch_gen", 0) + 1  # fresh table next run: clear the selection
            if go_foot:
                go_to(SEC_TRACK, obj=picked_id, view=VIEW_FOOT)
            else:
                go_to(SEC_TRIAGE, obj=picked_id, auto=True)
    st.download_button("Download CSV", data=table.assign(**{"NORAD ID": top["NORAD_CAT_ID"].astype(int)})
                       .assign(Level=levels.map({"red": "High", "yellow": "Medium", "green": "Low"}))
                       .to_csv(index=False).encode("utf-8"),
                       file_name="orbisk_watchlist.csv", mime="text/csv")
    if ca_meta is None or not top["ca_screened"].all():
        st.caption("Close-approach columns are filled for the highest-risk objects only (`tools/compute_conjunctions.py`); "
                   "— means not screened or no approach within the threshold.")
    # the globe comes last: the controls and the table are usable while it is still loading
    components.html(watchlist_globe_html(table_ids, tier_idx, tuple(float(r) for r in top["risk"]), deb_ids), height=660)

# ---------------------------------------------------------------- Tracker
@st.fragment
def tracker_view():
    picks = st.session_state.get("_picks", {}).pop(SEC_TRACK, None)  # set by "Ground re-entry footprint" in the watchlist
    if picks:
        st.session_state["tracker_obj"] = picks["obj"]
        st.session_state["tracker_view"] = picks.get("view", VIEW_ORBIT)
    norad = sticky(st.selectbox, "tracker_obj", "Object", ids, format_func=label)
    row = sats[sats["NORAD_CAT_ID"] == norad].iloc[0]
    record = records[int(norad)]
    name = record["OBJECT_NAME"]
    level = decay.traffic_light(sats["risk"]).loc[row.name]

    view = sticky(st.segmented_control, "tracker_view", "View", [VIEW_ORBIT, VIEW_DECAY, VIEW_FOOT], default=VIEW_ORBIT,
                  required=True, label_visibility="collapsed")
    when = now_key()  # globe, position and ground track share this minute, so their dots agree
    pos, track, propagate_error = tracker_geometry(int(norad), when)

    if view == VIEW_ORBIT:
        if propagate_error is not None:
            st.error(f"Could not propagate {name}: {propagate_error}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latitude", f"{pos['lat']:.2f}°")
            c2.metric("Longitude", f"{pos['lon']:.2f}°")
            c3.metric("Altitude", f"{pos['alt_km']:.0f} km")
            c4.metric("Perigee / Apogee", f"{row['perigee_km']:.0f} / {row['apogee_km']:.0f} km")
            components.html(tracker_globe_html(int(norad), when), height=610)  # starts paused at "now": matches the ground track

        # ---- summary card (below the globe)
        with st.container(border=True, key="summary_card"):
            h1, h2 = st.columns([4, 1])
            h1.markdown(f"#### {name}")
            h1.caption(f"NORAD ID {int(norad)} · {row['type'].capitalize()}")
            with h2.container(key=f"risk_badge_{level}"):  # the key drives the badge glow in the CSS
                st.badge({"red": "High risk", "yellow": "Medium risk", "green": "Low risk"}[level],
                         color={"red": "red", "yellow": "orange", "green": "green"}[level])
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Risk score", f"{row['risk']:.0f} / 100")
            s2.metric("Est. time to re-entry", decay.format_days(row["decay_days"]))
            s3.metric("Rough range", decay.format_range(row["decay_days_low"], row["decay_days_high"]))
            if bool(row.get("ca_screened", False)) and pd.notna(row.get("ca_miss_km")):
                s4.metric("Closest approach", f"{row['ca_miss_km']:.2f} km",
                          help=f"With {row['ca_with']}, {row['ca_hours']:.1f} h after the snapshot, at {row['ca_speed_km_s']:.1f} km/s.")
            else:
                s4.metric("Closest approach", "—", help="Not screened, or no approach inside the threshold.")
            if st.button("Screen this object", key="screen_this", type="primary",
                         help="Opens Conjunctions & Triage and screens this object for close approaches."):
                go_to(SEC_TRIAGE, obj=int(norad), auto=True)

    elif view == VIEW_DECAY:
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

    else:
        f1, f2 = st.columns(2)
        f1.metric("Latitudes overflown", f"±{footprint.max_latitude(row['INCLINATION']):.1f}°")
        f2.metric("Share of Earth's surface", f"{footprint.surface_fraction(row['INCLINATION']) * 100:.0f}%")
        st.plotly_chart(charts.footprint_figure(float(row["INCLINATION"])), width="stretch")
        if propagate_error is None:
            st.markdown("**Ground track** (next 3 orbits)")
            st.plotly_chart(globe.ground_track_figure(track, pos), width="stretch")
        st.caption("An object can only come down within the latitude band set by its inclination (red), and spends most of its time "
                   "near the band's edges (right). Where it actually re-enters depends on the final orbits and cannot be "
                   "predicted days ahead; this shows where it *can* pass, not where it *will* land.")

# ---------------------------------------------------------------- Triage desk
fmt_utc = lambda jd: conjunction.jd_to_datetime(jd).strftime("%Y-%m-%d %H:%M:%S")


def events_table(ev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Object": ev["name"], "NORAD ID": ev["norad_id"],
        "Closest approach (UTC)": [fmt_utc(j) for j in ev["tca_jd"]],
        "Miss distance (km)": ev["miss_km"], "Relative speed (km/s)": ev["rel_speed_km_s"],
        "Its perigee (km)": ev["other_perigee_km"], "Its apogee (km)": ev["other_apogee_km"],
        **({"Debris set": ev["other_group"].fillna("")} if "other_group" in ev else {}),
    })


@st.cache_data(ttl=300, show_spinner=False)
def conjunction_view_html(asset: int, partners: tuple, tcas: tuple, misses: tuple, start: str) -> str:
    """Globe with the screened satellite's orbit (cyan), each close partner's orbit and a marker where the pass happens.
    Orbit and marker colours follow the miss-distance bands of the cards."""
    from sgp4.api import jday
    t0 = orbits.TS.utc(*pd.Timestamp(start).timetuple()[:6])
    theta = np.radians(t0.gmst * 15.0) % (2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    tier_of = lambda miss: 0 if miss < 5 else (1 if miss < 15 else 2)
    paths = []
    for nid, tier in [(asset, 3)] + [(p, tier_of(m)) for p, m in zip(partners, misses)]:
        try:
            p = orbits.orbit_path_xyz(records[int(nid)], n_points=180, start=t0)
        except Exception:
            continue
        p["name"], p["tier"] = records[int(nid)]["OBJECT_NAME"], tier
        p["hover"] = records[int(nid)]["OBJECT_NAME"]
        paths.append(p)
    sat = conjunction.to_satrec(records[int(asset)])
    events = []
    for nid, tca, miss in zip(partners, tcas, misses):
        whole = np.floor(tca - 0.5) + 0.5  # sgp4 wants the Julian date split into day and fraction
        e, r, _ = sat.sgp4(float(whole), float(tca - whole))
        if e != 0:
            continue
        events.append({"x": r[0] * c + r[1] * s, "y": -r[0] * s + r[1] * c, "z": r[2], "tier": tier_of(miss),
                       "hover": f"{records[int(nid)]['OBJECT_NAME']}<br>{miss:.2f} km at {fmt_utc(tca)} UTC"})
    return globe.orbit_html(paths, height=560, events=events)


def miss_band(miss_km: float) -> tuple[str, str]:
    """(label, colour) for a miss distance: red under 5 km, yellow under 15 km, green otherwise."""
    if miss_km < 5:
        return "Under 5 km", "#e5484d"
    if miss_km < 15:
        return "5 to 15 km", "#e0a800"
    return "15 km or more", "#30a46c"


EVENT_CONFIG = {
    "NORAD ID": st.column_config.NumberColumn(format="%d"),
    "Miss distance (km)": st.column_config.NumberColumn(format="%.2f"),
    "Relative speed (km/s)": st.column_config.NumberColumn(format="%.2f"),
    "Its perigee (km)": st.column_config.NumberColumn(format="%.0f"),
    "Its apogee (km)": st.column_config.NumberColumn(format="%.0f"),
}


@st.fragment
def triage_view():
    picks = st.session_state.get("_picks", {}).pop(SEC_TRIAGE, None)  # set by "Screen this object" / "Screen for close approaches"
    if picks:
        st.session_state["tri_obj"] = picks["obj"]
    auto = bool(picks and picks.get("auto"))  # those buttons run the screening straight away
    st.subheader("Conjunctions & triage")
    st.markdown("Pick a satellite, a time window and a distance; Orbisk lists the closest passes.")
    asset_ids = ids  # the 50 highest-risk active satellites, same as the watchlist
    with st.form("tri_asset_form"):
        a1, a2, a3 = st.columns([3, 1, 1])
        asset = sticky(a1.selectbox, "tri_obj", "Object to screen", asset_ids, format_func=label)
        days = sticky(a2.slider, "tri_days", "Window (days)", min_value=1, max_value=30, value=2)
        hours = days * 24
        thr = sticky(a3.slider, "tri_thr", "Report within (km)", min_value=1, max_value=100, value=25)
        run = st.form_submit_button("Run screening", type="primary")
    if run or auto:
        jd0, ts0 = start_time("Snapshot time")
        with st.spinner("Propagating candidates…"):
            try:
                ev, n = triage.screen_asset(records[int(asset)], df, records, jd0, hours=hours, threshold_km=thr,
                                            against="all")
            except Exception as exc:
                st.error(f"Screening failed: {exc}")
            else:
                st.session_state["tri_asset"] = {"asset": int(asset), "events": ev, "n": n, "hours": ev.attrs.get("window_hours_effective", hours), "asked": hours, "thr": thr,
                                                 "start": ts0.strftime("%Y-%m-%d %H:%M")}
    res = st.session_state.get("tri_asset")
    if res:
        ev = res["events"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects in a similar altitude band", f"{res['n']:,}")
        m2.metric(f"Passes within {res['thr']} km", f"{len(ev)}")
        m3.metric("Closest pass", f"{ev['miss_km'].min():.1f} km" if len(ev) else "none")
        if res["hours"] < res["asked"] - 1:  # only shown when the screened object's orbit ends inside the window
            st.caption(f"This object's orbit ends after {res['hours'] / 24:.1f} days, so only that part was screened.")
        if not len(ev):
            st.success("No close approaches against any catalogue object inside the threshold in this window.")
        else:
            table_ev = events_table(ev)
            closest = ev.sort_values("miss_km").head(6)
            st.markdown("**Closest passes**")
            card_cols = st.columns(3)
            stripes = []
            for k, r in enumerate(closest.itertuples()):
                band = miss_band(r.miss_km)
                stripes.append(f".st-key-pass_card_{k} {{ border-left: 6px solid {band[1]} !important; }}")
                with card_cols[k % 3]:
                    with st.container(border=True, key=f"pass_card_{k}"):  # native bordered panel: follows the theme's border colour
                        st.markdown(
                            f'**{html.escape(str(r.name))}**  \n<span style="font-size:.85rem;opacity:.7">NORAD {int(r.norad_id)}</span>'
                            f'<div style="font-size:1.5rem;font-weight:600;color:{band[1]}">{r.miss_km:.2f} km</div>'
                            f'<span style="font-size:.85rem">{band[0]} · {fmt_utc(r.tca_jd)} UTC · {r.rel_speed_km_s:.1f} km/s</span>',
                            unsafe_allow_html=True)
            st.markdown("<style>" + "".join(stripes) + "</style>", unsafe_allow_html=True)
            if len(ev) > len(closest):
                st.caption(f"Showing the {len(closest)} closest of {len(ev)} passes; all of them are in the table below.")
            st.markdown("**Orbit view**")
            components.html(conjunction_view_html(int(res["asset"]), tuple(int(n) for n in closest["norad_id"]),
                                                  tuple(float(j) for j in closest["tca_jd"]),
                                                  tuple(float(m) for m in closest["miss_km"]), res["start"]), height=570)
            st.caption("Cyan: the screened satellite. ◆ marks each pass, coloured like the cards.")
            st.dataframe(table_ev, width="stretch", hide_index=True, column_config=EVENT_CONFIG)
            st.download_button("Download CSV", data=table_ev.to_csv(index=False).encode("utf-8"),
                               file_name=f"orbisk_conjunctions_{res['asset']}.csv", mime="text/csv", key="tri_csv")

# only the section on screen is built
if section == SEC_WATCH:
    watchlist_view()
elif section == SEC_TRACK:
    tracker_view()
else:
    triage_view()


# floating chat launcher (styles are in the CSS block at the top); the popover opens downward and closes with the same button
@st.fragment
def assistant_view():
    with st.container(key="ai_fab"):
        with st.popover("Ask Orbisk AI"):
            st.caption("Optional. The app is fully offline without this. When you add an Anthropic API key the assistant can "
                       "explain results in plain language. It never calculates orbits: it only sees numbers this app computed, "
                       "or looks data up through the app's own functions.")
            pasted = (st.text_input("Anthropic API key", type="password", key="pasted_key",
                                    help="Kept only in this browser session; not stored or logged.") or "").strip()
            configured = (_secret("ANTHROPIC_API_KEY") or "").strip()
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
                    max_req = int(_secret("ASSISTANT_MAX_REQUESTS") or ASSISTANT_DEFAULT_MAX)
                    used = st.session_state.setdefault("ai_requests", 0)
                    model = st.selectbox("Model", list(llm.MODELS), format_func=llm.MODELS.get, key="ai_model")
                    st.caption(f"{used}/{max_req} questions used in this session.")
                    ctx = llm.DataContext(df, records, components_df, weights, hist_all)

                    guarded = ai_guarded

                    t_chat, t_brief, t_risk, t_rec = st.tabs(["Ask the data", "Object briefing", "Risk explanation", "Recommended action"])
                    for tab, kind, task, button in ((t_brief, "brief", llm.BRIEFING_TASK, "Generate briefing"),
                                                    (t_risk, "risk", llm.RISK_TASK, "Explain risk score"),
                                                    (t_rec, "rec", llm.RECOMMEND_TASK, "Recommend next steps")):
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
                        st.caption("Examples: *Which red satellites have the closest approaches?* · *Why is STARLINK-1433 ranked so high?* · "
                                   "*What should we do about the ISS?* · *How is the risk score calculated?*")
                        chat_hist = st.session_state.setdefault("ai_chat", [])
                        for m in chat_hist:
                            with st.chat_message(m["role"]):
                                st.markdown(m["content"])
                                if m.get("trace"):
                                    st.caption("Looked up: " + ", ".join(f"{t['tool']}({t['input']})" for t in m["trace"]))
                        top_sat = sats.sort_values("risk", ascending=False).iloc[0]
                        suggestions = {  # button label -> question sent to the assistant
                            "Why is this object ranked first?": f"Why is {top_sat['OBJECT_NAME']} (NORAD {int(top_sat['NORAD_CAT_ID'])}) "
                                                               "ranked first by risk score?",
                            "Which satellites re-enter soonest?": "Which satellites have the soonest estimated re-entry?",
                            "What is perigee and apogee?": "What are perigee and apogee, and why do they matter for re-entry?",
                        }
                        asked = None
                        holder = st.empty()
                        if not chat_hist:
                            with holder.container():
                                for i, (label_, question) in enumerate(suggestions.items()):
                                    if st.button(label_, key=f"ai_suggest_{i}", width="stretch"):
                                        asked = question
                            if asked:
                                holder.empty()  # the chat has started: drop the suggestions
                        q = st.chat_input("Ask about the catalogue…") or asked
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
                            st.rerun(scope="fragment")


assistant_view()

with st.expander("How these numbers are computed"):
    st.markdown(
        "- **Perigee / apogee:** a from mean motion via Kepler's third law (n²a³ = μ); altitude = a(1∓e) − 6378.137 km.\n"
        "- **Positions and orbits:** SGP4 (via skyfield) on the stored mean elements; the globe is drawn in Earth-fixed axes with the Earth rotating at its real rate.\n"
        "- **Decay estimate:** the measured `MEAN_MOTION_DOT` (×2, since the field is ṅ/2) gives da/dt = −(2/3)·a·ṅ/n. "
        "That rate is scaled by air density (exponential atmosphere table) and integrated down to 120 km. "
        "Requires positive `BSTAR` (drag term). Uncertainty is at least a factor of 2.\n"
        "- **Risk score (0-100):** fixed weighted blend of seven terms scaled 0-1: perigee 20% (150 km = 1, 700 km or more = 0), "
        "decay time 20% (10 days or less = 1, 10 years or more = 0), inclination 10% (share of Earth's surface overflown), "
        "object type 5% (debris 1.0, rocket body 0.8, payload 0.3), miss distance 25% (closest approach to any satellite or debris in 24 h "
        "from the snapshot time: 1 km = 1, 25 km or more = 0), relative speed 10% (km/s over 15) and time to closest approach 10% "
        "(within 6 h = 1, 24 h or more = 0); the last two only count when the miss is under 25 km. Close-approach results are precomputed "
        "for the highest-risk objects.\n"
        "- **Colour levels:** by risk rank among the active satellites: the 10 highest-risk are red, the next 30 yellow, all others green "
        "(the watchlist draws on the 50 highest-risk satellites, so all 50 show 10 / 30 / 10; fewer are the first N of them in the chosen order). Debris is not on the watchlist; it can be drawn on the globe.\n"
        "- **Footprint:** latitude band ±min(i, 180−i); time per latitude from sin φ = sin i · sin u.\n"
        "- **Conjunctions:** coarse-grid SGP4 screening with 0.5 s refinement of local minima, against every satellite and debris fragment.\n"
        "- **Scope:** low Earth orbit (LEO) only: the watchlist, tracker, screening and assistant cover active satellites with a mean altitude and perigee below 2,000 km. "
        "Objects in higher orbits are not listed (a higher orbit that crosses LEO still counts as a close-approach partner).\n"
        "- **Debris:** the debris groups (TLE or CelesTrak JSON files in data/debris) are parsed into the same record format. They are not on the watchlist "
        "or in the pickers; they are only drawn as orbits on the Watchlist globe and used as close-approach partners when an active satellite is screened "
        "(Conjunctions & Triage, and the close-approach terms of the risk score). Altitude and risk figures are for active satellites only."
        "Altitude bands are the densest ranges of the active catalogue, found from the data.\n"
        "- **Assistant (optional):** off unless an API key is supplied. The model only receives numbers computed above, or calls the app's own lookup functions; it never computes orbits.\n"
        "- **Limits:** one element set per object (no history until snapshots are added), mean elements are only km-accurate, "
        "and active satellites that manoeuvre invalidate drag-based decay estimates."
    )

# make the launcher draggable: script runs in a same-origin iframe and drives the parent page; the position is kept in
# a <style> tag in the parent head (survives Streamlit re-renders) and in localStorage
components.html("""<script>
const P = window.parent, D = P.document, KEY = "orbisk_fab_pos_v2";  // v2: drops positions saved by earlier versions
if (!P.__orbiskDrag) {
  P.__orbiskDrag = true;
  const style = D.createElement("style");
  D.head.appendChild(style);
  const nameButton = () => {
    const b = D.querySelector(".st-key-ai_fab button");
    if (b && b.getAttribute("aria-label") !== "Ask Orbisk AI") b.setAttribute("aria-label", "Ask Orbisk AI");
  };
  nameButton();
  new MutationObserver(nameButton).observe(D.body, {childList: true, subtree: true});
  const apply = (x, y) => {
    const w = 56, h = 56;
    x = Math.min(Math.max(24, x), Math.max(24, P.innerWidth - w - 24)); y = Math.min(Math.max(12, y), Math.max(12, P.innerHeight - h - 30));  
    style.textContent = `.st-key-ai_fab{left:${x}px !important;top:${y}px !important;right:auto !important;}`;
    return [x, y];
  };
  const reset = () => { style.textContent = ""; try { P.localStorage.removeItem(KEY); } catch (e) {} };
  const restore = () => {
    try {
      const s = JSON.parse(P.localStorage.getItem(KEY));
      if (s && Number.isFinite(s[0]) && Number.isFinite(s[1]) && P.innerWidth > 200 && P.innerHeight > 200) apply(s[0], s[1]);
    } catch (e) {}
  };
  // if the button ever ends up outside the window (saved position, resized window), put it back in its default corner
  const rescue = () => {
    const b = D.querySelector(".st-key-ai_fab button");
    if (!b) return;
    const r = b.getBoundingClientRect();
    if (r.width > 0 && (r.right < 20 || r.bottom < 20 || r.left > P.innerWidth - 20 || r.top > P.innerHeight - 20)) reset();
  };
  restore();
  setTimeout(rescue, 1500); setTimeout(rescue, 5000);
  D.addEventListener("dblclick", (e) => { if (e.target.closest(".st-key-ai_fab")) reset(); }, true);
  let drag = null;
  D.addEventListener("pointerdown", (e) => {
    const fab = e.target.closest(".st-key-ai_fab");
    if (!fab || e.button > 0) return;
    const r = fab.getBoundingClientRect();
    drag = {dx: e.clientX - r.left, dy: e.clientY - r.top, x0: e.clientX, y0: e.clientY, moved: false};
  }, true);
  D.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 6) return;
    drag.moved = true;
    drag.pos = apply(e.clientX - drag.dx, e.clientY - drag.dy);
    e.preventDefault();
  }, true);
  D.addEventListener("pointerup", () => {
    if (drag && drag.moved) {
      try { P.localStorage.setItem(KEY, JSON.stringify(drag.pos)); } catch (e) {}
      const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };  // a drag must not open the popup
      D.addEventListener("click", swallow, {capture: true, once: true});
      setTimeout(() => D.removeEventListener("click", swallow, true), 300);
    }
    drag = null;
  }, true);
  // turning the "Show debris orbits" toggle off hides the debris in the globe straight away; the new page then
  // arrives after the (slow) rerun and keeps it hidden
  D.addEventListener("click", (e) => {
    const box = e.target.closest('[data-testid="stCheckbox"], [data-testid="stToggle"]');
    if (!box || !box.textContent.includes("Show debris orbits on globe")) return;
    setTimeout(() => {
      const input = box.querySelector("input");
      if (!input || input.checked) return;
      for (const f of D.querySelectorAll("iframe")) {
        try {
          for (const l of f.contentDocument.querySelectorAll("label")) {
            const cb = l.querySelector('input[type="checkbox"]');
            if (cb && cb.checked && /Debris orbits \\(/.test(l.textContent)) cb.click();
          }
        } catch (err) {}
      }
    }, 50);
  }, true);
  P.addEventListener("resize", () => { restore(); setTimeout(rescue, 300); });
}
</script>""", height=0)
