import sys, json, time
sys.path.insert(0, sys.argv[1])
import dataset, decay, history, llm
cat, recs = dataset.load_catalogue()
df = dataset.attach_closest_approaches(cat, dataset.load_closest_approaches())
ctx = llm.DataContext(df, recs, decay.risk_components(df), dict(decay.DEFAULT_WEIGHTS), history.load_history())
client = llm.make_client()
qs = ["When is ZHUQUE-2E R/B expected to re-enter, and how uncertain is that estimate?",
      "Which watchlist satellites have the closest approaches in the next 24 hours?",
      "Screen STARLINK-4425 for close approaches within 10 km over the next 6 hours.",
      "Which latitudes could STARLINK-2635 come down over, and what share of Earth's surface is that?"]
out = []
for q in qs:
    try:
        answer, trace, usage = llm.chat(client, llm.DEFAULT_MODEL, ctx, [], q)
    except Exception as e:
        print("ERROR", type(e).__name__, str(e)[:200]); break
    tools = [(t["tool"], t["input"]) for t in trace]
    outputs = [llm.run_tool(ctx, n, a)[0] for n, a in tools]
    out.append(dict(q=q, answer=answer, tools=tools, tool_outputs=outputs, usage=usage))
    print("ok", q, "| tools:", [t[0] for t in tools], "| usage:", usage, flush=True)
json.dump(out, open("results/e4b_run.json", "w"), indent=1, default=str)
