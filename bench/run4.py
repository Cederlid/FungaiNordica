"""Ask every bench case through the server's chat endpoint, as LibreChat does (streamed).

OUT=<file> python3 run4.py   -- resumes: cases already answered without error are skipped
"""
import json, os, sys, time
import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("OUT", "results6.jsonl"))
URL = "http://127.0.0.1:8000/v1/chat/completions"
cases = json.load(open(os.path.join(HERE, os.environ.get("CASES", "all_cases.json"))))
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if not r["error"]:
            done.add(r["i"])


def ask(messages):
    start, first, text = time.time(), None, []
    with httpx.stream("POST", URL, json={"model": "qwen3.8-27b", "messages": messages, "stream": True},
                      timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if piece:
                first = first or time.time() - start
                text.append(piece)
    answer = "".join(text)
    if "did not answer this time" in answer:
        raise RuntimeError(answer[-200:])
    return answer, [first or 0.0, time.time() - start]


ONLY = {int(x) for x in os.environ.get("ONLY", "").split(",") if x}
for i, case in enumerate(cases):
    if i in done or (ONLY and i not in ONLY):
        continue
    turns = case.get("turns") or [case["query"]]
    messages, answers, times, error = [], [], [], ""
    try:
        for turn in turns:
            messages.append({"role": "user", "content": turn})
            answer, t = ask(messages)
            messages.append({"role": "assistant", "content": answer})
            answers.append(answer); times.append(t)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps({"i": i, "answers": answers, "answer": answers[-1] if answers else "",
                            "times": times, "error": error}, ensure_ascii=False) + "\n")
    t = times[-1] if times else [0, 0]
    print(f"{i} {case['kind']:12} first={t[0]:.0f}s total={t[1]:.0f}s {error[:80]}", flush=True)
print("done")
