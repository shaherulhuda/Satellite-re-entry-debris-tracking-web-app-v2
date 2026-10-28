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
    leo = dict(zip(cat.NORAD_CAT_ID, cat.leo))
    assert out["total_matches"] > 0 and all(src[o["norad_id"]] == "active" and leo[o["norad_id"]] for o in out["objects"])
    for tool, args in (("get_object_details", {"norad_id": deb_id}), ("screen_conjunctions", {"norad_id": deb_id})):
        text, is_err = llm.run_tool(ctx, tool, args)
        assert "debris" in text.lower() or "not an active" in text
    assert json.loads(llm.run_tool(ctx, "get_object_details", {"norad_id": 25544})[0])["norad_id"] == 25544


def test_risk_rank_is_among_satellites_and_screening_still_uses_debris(world):
    cat, recs, ctx = world
    facts = llm.object_facts(ctx, 25544, include_position=False)
    assert facts["risk"]["out_of"] == int(((cat.source == "active") & cat.leo).sum())
    out = json.loads(llm.run_tool(ctx, "screen_conjunctions", {"norad_id": 25544, "hours": 6, "threshold_km": 50})[0])
    assert out["candidates_screened"] > 0 and "approaches" in out


def test_app_lists_and_pickers_are_satellites_only():
    from streamlit.testing.v1 import AppTest
    cat, _ = dataset.load_catalogue()
    deb = set(cat[cat.source == "debris"].NORAD_CAT_ID)
    at = AppTest.from_file("app.py", default_timeout=400).run()
    assert not at.exception
    table = at.dataframe[0].value
    deb_names = set(cat[cat.source == "debris"].OBJECT_NAME)
    assert not table["Name"].str.contains("DEB").any() and not set(table["Name"]) & deb_names
    # the table fits on screen: no NORAD ID, Type or With columns, and the risk bar is present
    assert not {"NORAD ID", "Type", "With"} & set(table.columns) and "Risk score" in table.columns
    # pool = the 50 highest-risk active satellites (also what the watchlist draws on)
    full = dataset.attach_closest_approaches(cat, dataset.load_closest_approaches())
    full["risk"] = decay.risk_score(decay.risk_components(full))
    pool = full[full.source == "active"].sort_values("risk", ascending=False).head(50).NORAD_CAT_ID.astype(int).tolist()
    for box in at.selectbox:
        if box.label in ("Object", "Object to screen"):
            ids = [int(re.search(r"\((\d+)\)$", o).group(1)) for o in box.options]
            assert ids == pool and not set(ids) & deb  # same 50 satellites, highest risk first
    assert not any("Follow a debris object" in str(r.options) for r in at.radio)  # debris is never the subject
    # debris only enters through the globe box and the screening partners
    assert [t.label for t in at.toggle] == ["Show debris orbits on globe"] and not [s for s in at.slider if s.label == "Debris orbits to draw"]  # slider only when toggled on
    at.toggle[0].set_value(True).run()
    assert not at.exception and [s.label for s in at.slider if s.label == "Debris orbits to draw"] == ["Debris orbits to draw"]
    assert next(s for s in at.slider if s.label == "Debris orbits to draw").value == 50  # starts at 50


def test_assistant_is_leo_only(world):
    cat, recs, ctx = world
    non_leo = cat[(cat.source == "active") & ~cat.leo]
    assert len(non_leo) > 0  # GEO/MEO satellites exist in the catalogue
    nid = int(non_leo.NORAD_CAT_ID.iloc[0])
    for tool in ("get_object_details", "screen_conjunctions"):
        text, _ = llm.run_tool(ctx, tool, {"norad_id": nid})
        assert "low-Earth-orbit" in text
    out = json.loads(llm.run_tool(ctx, "search_objects", {"min_perigee_km": 2000, "limit": 5})[0])
    assert out["total_matches"] == 0


def test_levels_and_closest_approach_in_search_and_facts(world):
    cat, recs, ctx = world
    red = json.loads(llm.run_tool(ctx, "search_objects", {"level": "red", "limit": 25})[0])
    assert red["total_matches"] == 10 and all(o["level"] == "red" for o in red["objects"])
    facts = llm.object_facts(ctx, red["objects"][0]["norad_id"], include_position=False)
    assert facts["risk"]["level"] == "red" and facts["suggested_action"]["level"] in ("prepare", "act")


def test_suggested_action_rules():
    f = llm.suggested_action
    assert f("green", 2.0, 5, 900)["level"] == "act"
    assert f("green", 2.0, 40, 900)["level"] == "prepare"
    assert f("green", None, None, 12)["level"] == "prepare"
    assert f("red", None, None, float("nan"))["level"] == "prepare"
    assert f("green", None, None, 900)["level"] == "monitor"


def test_object_page_card_subtabs_and_screen_button():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=400).run()
    assert not at.exception
    assert [t.label for t in at.tabs if t.label in ("Orbit", "Decay", "Re-entry footprint", "Risk breakdown")] == \
        ["Orbit", "Decay", "Re-entry footprint", "Risk breakdown"]
    assert any(m.label == "Risk score" for m in at.metric)  # summary card
    pick = at.selectbox(key="tracker_obj")
    target = pick.options[3]  # option strings look like "NAME (12345)"
    pick.select_index(3).run()
    assert not at.exception
    at.button(key="screen_this").click().run()
    assert not at.exception
    assert str(at.session_state["tri_obj"]) == re.search(r"\((\d+)\)$", target).group(1)
    assert str(at.selectbox(key="tri_obj").value) == re.search(r"\((\d+)\)$", target).group(1)


def test_assistant_has_suggested_question_buttons(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    at = AppTest.from_file("app.py", default_timeout=400).run()
    assert not at.exception
    labels = [b.label for b in at.button if b.key and b.key.startswith("ai_suggest_")]
    assert labels == ["Why is this object ranked first?", "Which satellites re-enter soonest?", "What is perigee and apogee?"]


def test_clicking_a_suggestion_asks_the_assistant(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import llm as llm_mod
    seen = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm_mod, "make_client", lambda key=None: object())
    monkeypatch.setattr(llm_mod, "chat", lambda c, model, ctx, hist, q: (seen.append(q), ("ok", [], {"input_tokens": 1, "output_tokens": 1}))[1])
    at = AppTest.from_file("app.py", default_timeout=400).run()
    at.button(key="ai_suggest_1").click().run()
    assert not at.exception
    assert seen == ["Which satellites have the soonest estimated re-entry?"]
    assert not [b for b in at.button if b.key and b.key.startswith("ai_suggest_")]  # hidden once the chat has started


def test_triage_cards_and_bands():
    import ast
    src = open("app.py").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "miss_band")
    ns = {}
    exec(compile(ast.Module([fn], []), "app.py", "exec"), ns)
    mb = ns["miss_band"]
    assert [mb(x)[1] for x in (0.5, 4.99, 5.0, 14.9, 15.0, 80)] == \
        ["#e5484d", "#e5484d", "#e0a800", "#e0a800", "#30a46c", "#30a46c"]


def test_triage_shows_intro_then_cards_table_and_csv():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=600).run()
    assert any(m.value == "Pick a satellite, a time window and a distance; Orbisk lists the closest passes." for m in at.markdown)
    next(b for b in at.button if b.label == "Run screening").click()
    at.run()
    assert not at.exception
    stripes = [m.value for m in at.markdown if ".st-key-pass_card_0" in m.value]
    assert stripes and stripes[0].count("border-left: 6px solid") <= 6
    assert all("rgba(128,128,128" not in m.value for m in at.markdown if "NORAD" in m.value or "pass_card" in m.value)  # panels are native, no hard-coded grey
    assert at.button(key="FormSubmitter:tri_asset_form-Run screening").proto.type == "primary"
