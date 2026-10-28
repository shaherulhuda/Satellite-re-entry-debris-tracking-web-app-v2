"""Ask the assistant a fixed question set and check each answer for the facts the app computed.

Needs a real key:  ANTHROPIC_API_KEY=... python tools/check_assistant.py [--model claude-haiku-4-5]
Each question has keywords the answer must contain (taken from the app's own data, not from the model).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset  # noqa: E402
import decay  # noqa: E402
import llm  # noqa: E402


def build_context():
    cat, recs = dataset.load_catalogue()
    return llm.DataContext(cat, recs, decay.risk_components(cat))


def questions(ctx):
    sats = ctx.satellites().sort_values("risk", ascending=False)
    top = sats.iloc[0]
    return [
        ("What is the highest-risk satellite and its score?", [top["OBJECT_NAME"], str(int(round(top["risk"])))]),
        ("How is the risk score calculated?", ["perigee", "miss distance", "25%"]),
        ("Does this app cover geostationary satellites?", ["2,000", "low"]),
        (f"What should we do about {top['OBJECT_NAME']}?", [top["OBJECT_NAME"], "monitor|prepare|act"]),
        ("What does the Re-entry Tracker show?", ["footprint|ground track", "decay"]),
        ("How many satellites are red?", ["10"]),
    ]


def check(answer: str, keywords) -> list[str]:
    low = answer.lower()
    return [k for k in keywords if not any(alt.lower() in low for alt in k.split("|"))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    args = p.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    ctx, client, failed = build_context(), llm.make_client(), 0
    for q, keywords in questions(ctx):
        answer, trace, usage = llm.chat(client, args.model, ctx, [], q)
        missing = check(answer, keywords)
        failed += bool(missing)
        print(f"{'FAIL' if missing else 'ok  '} {q}\n     tools: {[t['tool'] for t in trace]}  missing: {missing}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
