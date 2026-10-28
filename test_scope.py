"""Scope rule: the watchlist, tracker and screening pickers are active satellites only (active.json).
Debris is only drawn on the globe and used as a close-approach partner."""
import json
import re

import numpy as np
import pytest

import dataset
import decay
import llm


@pytest.fixture(scope="module")
def world():
    cat, recs = dataset.load_catalogue()
    return cat, recs, llm.DataContext(cat, recs, decay.risk_components(cat))


def test_assistant_tools_only_see_active_satellites(world):
    cat, recs, ctx = world
    deb_id = int(cat[cat.source == "debris"].NORAD_CAT_ID.iloc[0])
    out = json.loads(llm.run_tool(ctx, "search_objects", {"name_contains": "DEB", "limit": 5})[0])
    assert out["total_matches"] == 0 or all("DEB" not in o["name"] for o in out["objects"])
    out = json.loads(llm.run_tool(ctx, "search_objects", {"max_perigee_km": 400, "limit": 25})[0])
    src = dict(zip(cat.NORAD_CAT_ID, cat.source))
    assert out["total_matches"] > 0 and all(src[o["norad_id"]] == "active" for o in out["objects"])
    for tool, args in (("get_object_details", {"norad_id": deb_id}), ("screen_conjunctions", {"norad_id": deb_id})):
        text, is_err = llm.run_tool(ctx, tool, args)
        assert "debris" in text.lower() or "not an active satellite" in text
    assert json.loads(llm.run_tool(ctx, "get_object_details", {"norad_id": 25544})[0])["norad_id"] == 25544


def test_risk_rank_is_among_satellites_and_screening_still_uses_debris(world):
    cat, recs, ctx = world
    facts = llm.object_facts(ctx, 25544, include_position=False)
    assert facts["risk"]["out_of"] == int((cat.source == "active").sum())
    out = json.loads(llm.run_tool(ctx, "screen_conjunctions", {"norad_id": 25544, "hours": 6, "threshold_km": 50})[0])
    assert out["candidates_screened"] > 0 and "approaches" in out


def test_app_lists_and_pickers_are_satellites_only():
    from streamlit.testing.v1 import AppTest
    cat, _ = dataset.load_catalogue()
    deb = set(cat[cat.source == "debris"].NORAD_CAT_ID)
    at = AppTest.from_file("app.py", default_timeout=400).run()
    assert not at.exception
    table = at.dataframe[0].value
    assert not table["Name"].str.contains("DEB").any() and not table["Type"].eq("debris").any()
    for box in at.selectbox:
        if box.label in ("Object", "Object to screen"):
            ids = [int(re.search(r"\((\d+)\)$", o).group(1)) for o in box.options]
            assert ids and not set(ids) & deb
    assert not any("Follow a debris object" in str(r.options) for r in at.radio)  # debris is never the subject
    # debris only enters through the globe box and the screening partners
    assert [n.label for n in at.number_input if n.label == "Debris"] == ["Debris"]
