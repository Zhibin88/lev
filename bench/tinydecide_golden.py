"""golden/tinydecide from TinyDecide's own conformance suite (conformance/ in TheREZOR/TinyDecide,
the outputs of its JavaScript reference engine).

    python bench/tinydecide_golden.py <TinyDecide dir> golden/tinydecide

tok.edn: every tokenizer case with its ids. requests.edn: the requests lev can ask (span questions
dropped: a question attends to the state and itself only, so the others' answers do not move), with
the reference's answers for the 10.4M build, and its ids when nothing was dropped; requests.full.edn
the same for the 13.8M build (full/). errors.edn: the requests the reference refuses, minus the span
ones.
"""
import json
import sys
from pathlib import Path


def edn(v):
    if v is None:
        return "nil"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=True)
    if isinstance(v, list):
        return "[" + " ".join(edn(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + " ".join(f":{k} {edn(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def answer(a):
    if a["type"] == "noul":
        return {"type": "noul", "p": a["p"]}
    out = {"type": a["type"], "probs": a["probs"], "pick": a["pick"], "confidence": a["confidence"]}
    if a["type"] == "score":
        out["score"] = a["score"]
    return out


def requests(cases, exp):
    reqs, errs = [], []
    for r, x in zip(cases["requests"], exp["requests"]):
        qs = r["questions"]
        if "error" in x:
            if all(q["type"] != "span" for q in qs):
                errs.append({"name": r["name"], "state": r["state"], "questions": qs, "error": x["error"]})
            continue
        keep = [i for i, q in enumerate(qs) if q["type"] != "span"]
        if not keep:
            continue
        e = {"name": r["name"], "state": r["state"], "questions": [qs[i] for i in keep],
             "answers": [answer(x["answers"][i]) for i in keep], "truncated": x["truncated"]}
        if len(keep) == len(qs):
            e["ids"] = x["ids"]
        reqs.append(e)
    return reqs, errs


def main(src, dst):
    conf = Path(src) / "conformance"
    cases = json.loads((conf / "cases.json").read_text(encoding="utf-8"))
    exp = json.loads((conf / "expected.S768.json").read_text(encoding="utf-8"))
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    tok = [{"text": t, "ids": ids} for t, ids in zip(cases["tokenizer"], exp["tokenizer"])]
    (out / "tok.edn").write_text("[" + "\n ".join(edn(x) for x in tok) + "]\n", encoding="ascii")
    reqs, errs = requests(cases, exp)
    full, _ = requests(cases, json.loads((conf / "expected.full.json").read_text(encoding="utf-8")))
    (out / "requests.full.edn").write_text("[" + "\n ".join(edn(x) for x in full) + "]\n", encoding="ascii")
    (out / "requests.edn").write_text("[" + "\n ".join(edn(x) for x in reqs) + "]\n", encoding="ascii")
    (out / "errors.edn").write_text("[" + "\n ".join(edn(x) for x in errs) + "]\n", encoding="ascii")
    print(f"{len(tok)} tokenizer cases, {len(reqs)} requests, {len(errs)} errors -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
