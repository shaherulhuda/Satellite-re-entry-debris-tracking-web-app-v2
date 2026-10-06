import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

import decay
import history
import llm
import orbits


@pytest.fixture(scope="module")
def ctx():
    cat = decay.estimate_decay(orbits.load_catalog())
    cat["type"] = cat["OBJECT_NAME"].map(decay.classify_type)
    return llm.DataContext(cat, {r["NORAD_CAT_ID"]: r for r in orbits.load_records()},
                           decay.risk_components(cat), history=history.load_history())


def test_importing_llm_does_not_import_anthropic():
    code = "import sys, llm; assert 'anthropic' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_search_matches_pandas_and_caps_rows(ctx):
    out, err = llm.run_tool(ctx, "search_objects", {"max_perigee_km": 200, "min_inclination_deg": 50, "limit": 100})
    d = json.loads(out)
    cat = ctx.catalogue
    expected = int(((cat.perigee_km <= 200) & (cat.INCLINATION >= 50)).sum())
    assert not err and d["total_matches"] == expected and d["returned"] <= llm.MAX_ROWS
    assert all(o["perigee_km"] <= 200 and o["inclination_deg"] >= 50 for o in d["objects"])


def test_details_by_name_and_ambiguity(ctx):
    d = json.loads(llm.run_tool(ctx, "get_object_details", {"name": "ISS (ZARYA)"})[0])
    assert d["norad_id"] == 25544 and d["footprint"]["max_latitude_deg"] == pytest.approx(51.6, abs=0.1)
    assert abs(sum(d["risk"]["weights"].values()) - 1.0) < 0.02
    amb = json.loads(llm.run_tool(ctx, "get_object_details", {"name": "STARLINK"})[0])
    assert amb["ambiguous"] and len(amb["matches"]) <= 5


def test_tool_errors_are_returned_not_raised(ctx):
    assert llm.run_tool(ctx, "nope", {})[1] is True
    out, err = llm.run_tool(ctx, "get_object_details", {"norad_id": 1})
    assert err is True and "error" in json.loads(out)
    assert llm.run_tool(ctx, "screen_conjunctions", {"norad_id": 99999999})[1] is False  # reports error as data


def test_screen_tool(ctx):
    d = json.loads(llm.run_tool(ctx, "screen_conjunctions", {"norad_id": 25544, "hours": 2, "threshold_km": 50})[0])
    assert d["candidates_screened"] > 0 and "approaches" in d


class FakeClient:
    """Plays back scripted responses and records the requests it received."""
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def msg(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=10, output_tokens=5))


def text(t):
    return NS(type="text", text=t)


def tool(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def test_chat_runs_tool_loop(ctx):
    fake = FakeClient([
        msg("tool_use", tool("t1", "search_objects", {"max_perigee_km": 200, "limit": 3})),
        msg("end_turn", text("Found some.")),
    ])
    answer, trace, usage = llm.chat(fake, "claude-opus-5-5", ctx, [], "what is below 200 km?")
    assert answer == "Found some." and trace[0]["tool"] == "search_objects"
    assert usage == {"input_tokens": 20, "output_tokens": 10}
    second = fake.calls[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"][0]["tool_use_id"] == "t1"
    assert json.loads(second[-1]["content"][0]["content"])["total_matches"] > 0
    assert fake.calls[0]["output_config"] == {"effort": "medium"} and "tool_choice" not in fake.calls[0]


def test_chat_step_cap_and_haiku_has_no_effort(ctx):
    fake = FakeClient([msg("tool_use", tool(f"t{i}", "search_objects", {})) for i in range(3)])
    answer, trace, _ = llm.chat(fake, "claude-haiku-4-5", ctx, [], "loop forever", max_steps=3)
    assert "too many tool calls" in answer and len(trace) == 3
    assert all("output_config" not in c for c in fake.calls)


def test_explain_sends_only_app_facts(ctx):
    facts = llm.object_facts(ctx, 25544)
    fake = FakeClient([msg("end_turn", text("Brief."))])
    out, usage = llm.explain(fake, "claude-opus-5-5", facts, llm.BRIEFING_TASK)
    sent = fake.calls[0]["messages"][0]["content"]
    assert out == "Brief." and '"norad_id": 25544' in sent and "tools" not in fake.calls[0]
