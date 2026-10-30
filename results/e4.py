import sys, os, json, time
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tools")
import check_assistant as ca, llm
ctx, client = ca.build_context(), llm.make_client()
out = []; tin = tout = 0
for qtext, kw in ca.questions(ctx):
    t0 = time.time()
    answer, trace, usage = llm.chat(client, llm.DEFAULT_MODEL, ctx, [], qtext)
    missing = ca.check(answer, kw)
    u = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
    out.append(dict(q=qtext, keywords=kw, missing=missing, tools=[t["tool"] for t in trace], usage=u,
                    seconds=round(time.time() - t0, 1), answer=answer, trace=trace))
    print(("FAIL" if missing else "ok  "), qtext, "| tools:", [t["tool"] for t in trace], "| usage:", u, flush=True)
json.dump(out, open("results/e4_run.json", "w"), indent=1, default=str)
