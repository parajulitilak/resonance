"""The inspection agent, its independent verifier, and the engineer's approval gate.

Three layers of checking, outermost last:
1. physics: the harmonic-ladder verifier in `spectrum.py` decides which takes are measurable;
2. code: hard rules applied to the agent's report (no number without a verified ladder,
   no absolute field tension, no "safe");
3. model: Muse Spark Open re-reads only the numbers and must agree with every claim;
   anything it disputes is marked "needs engineer".
Nothing is final until an engineer approves it (`approve`).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .llm import chat

BANNED = re.compile(r"\b(safe|unsafe|guaranteed|certif\w*)\b", re.I)

INSPECTOR_SYSTEM = """You are Resonance's inspection agent. You review vibration measurements of cables, \
ropes or bridge hangers taken from a phone video and prepare a report for a human engineer.

Rules:
1. Use only numbers returned by tools, and cite the field you used (for example `ladder.f0_hz`).
2. If a member's harmonic ladder is not verified (`ladder.ok` is false), make no frequency or tension \
claim for it. Request a re-shoot with concrete instructions (frame rate, distance, viewing angle, how to \
excite the member, lighting), based on the refusal reasons.
3. In field mode, never state the absolute tension of a member. Compare members with each other and with \
design values given in the context, and flag outliers for inspection.
4. In bench mode, report the measured tension with its uncertainty and compare it with any reference \
values given in the context.
5. Never call anything safe or unsafe. Use flags with severity info, inspect or urgent for the engineer.
6. Keep statements short and factual. Call submit_report exactly once, at the end."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_members", "description": "One-line summary of every measured member.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_member", "description": "Full measurement evidence for one member.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "compare_members",
        "description": "Robust outlier statistics across members with verified ladders (f0*L if lengths known, else f0).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "request_reshoot", "description": "Ask the field team to film a member again.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "reason": {"type": "string"}, "instructions": {"type": "string"}},
            "required": ["name", "reason", "instructions"]}}},
    {"type": "function", "function": {
        "name": "flag", "description": "Flag a finding for the engineer.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "severity": {"type": "string", "enum": ["info", "inspect", "urgent"]},
            "finding": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "severity", "finding", "evidence"]}}},
    {"type": "function", "function": {
        "name": "submit_report", "description": "Submit the final report.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "members": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "status": {"type": "string", "enum": ["measured", "refused", "flagged"]},
                "statement": {"type": "string"}}, "required": ["name", "status", "statement"]}},
            "next_steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary", "members", "next_steps"]}}},
]


def _member_summary(m: dict) -> dict:
    lad, ten = m["ladder"], m.get("tension") or {}
    return {"name": m["name"], "ladder_ok": lad["ok"], "f0_hz": None if not lad["ok"] else round(lad["f0_hz"], 4),
            "modes": [x["n"] for x in lad["modes"]], "refusal": lad["reasons"],
            "tension_n": None if ten.get("value_n") is None else round(ten["value_n"], 2),
            "tension_sigma_n": None if ten.get("sigma_n") is None else round(ten["sigma_n"], 2),
            "noise_px": round(m["quality"]["noise_px"], 4)}


def compare(report: dict) -> dict:
    rows = []
    for m in report["members"]:
        if not m["ladder"]["ok"]:
            continue
        L = ((m.get("tension") or {}).get("inputs") or {}).get("length_m")
        rows.append((m["name"], m["ladder"]["f0_hz"] * L if L else m["ladder"]["f0_hz"], bool(L)))
    if len(rows) < 3:
        return {"note": "fewer than 3 verified members: no outlier statistics", "members": [r[0] for r in rows]}
    v = np.array([r[1] for r in rows])
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) or 1e-9
    z = 0.6745 * (v - med) / mad
    return {"quantity": "f0*L (proportional to sqrt(T/m))" if all(r[2] for r in rows) else "f0",
            "median": med, "members": [{"name": r[0], "value": float(x), "robust_z": float(zz),
                                        "outlier": bool(abs(zz) > 3.5)} for r, x, zz in zip(rows, v, z)]}


def run_inspector(report: dict, context: str, mode: str = "bench", provider: str = "baseten",
                  max_turns: int = 14, log=print) -> dict:
    by = {m["name"]: m for m in report["members"]}
    actions: dict = {"reshoots": [], "flags": [], "report": None, "transcript": [], "usage": []}

    def call(name: str, args: dict):
        if name == "list_members":
            return [_member_summary(m) for m in report["members"]]
        if name == "get_member":
            m = by.get(args.get("name"))
            return m if m else {"error": f"no member {args.get('name')!r}"}
        if name == "compare_members":
            return compare(report)
        if name == "request_reshoot":
            actions["reshoots"].append(args)
            return {"recorded": True}
        if name == "flag":
            actions["flags"].append(args)
            return {"recorded": True}
        if name == "submit_report":
            actions["report"] = args
            return {"recorded": True}
        return {"error": f"unknown tool {name}"}

    msgs = [{"role": "system", "content": INSPECTOR_SYSTEM},
            {"role": "user", "content": f"Mode: {mode}.\nContext from the field team:\n{context}\n\n"
                                        f"Video timing: {json.dumps(report['timing'])}\n"
                                        "Review every member, then submit the report."}]
    for turn in range(max_turns):
        r = chat(msgs, tools=TOOLS, provider=provider)
        actions["usage"].append(r["usage"])
        msg = r["message"]
        calls = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                    else {"role": "assistant", "content": msg.get("content") or ""})
        actions["transcript"].append({"turn": turn, "content": msg.get("content"),
                                      "tool_calls": [c["function"] for c in calls]})
        if not calls:
            if actions["report"]:
                break
            msgs.append({"role": "user", "content": "Use the tools; finish with submit_report."})
            continue
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            res = call(fn, args)
            log(f"  agent → {fn}({', '.join(f'{k}={v!r}' for k, v in args.items())[:120]})")
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res, default=float)[:12000]})
        if actions["report"]:
            break
    return actions


def enforce_rules(report: dict, actions: dict, mode: str) -> list[dict]:
    """Code-level checks on the agent's output. Returns violations (and fixes statuses in place)."""
    out = []
    by = {m["name"]: m for m in report["members"]}
    rep = actions.get("report") or {"members": [], "summary": "", "next_steps": []}
    for item in rep["members"]:
        m = by.get(item["name"])
        if m is None:
            out.append({"member": item["name"], "rule": "unknown member"})
            continue
        if not m["ladder"]["ok"] and item["status"] != "refused":
            out.append({"member": item["name"], "rule": "claim on unverified ladder → forced to refused"})
            item["status"] = "refused"
        if mode == "field" and re.search(r"\d\s*(k?N)\b", item["statement"]):
            out.append({"member": item["name"], "rule": "absolute tension in field mode → needs engineer"})
            item["status"] = "flagged"
    texts = [rep.get("summary", "")] + [i["statement"] for i in rep["members"]] + \
            [f["finding"] for f in actions["flags"]]
    for t in texts:
        if BANNED.search(t or ""):
            out.append({"member": "-", "rule": f"banned wording: {BANNED.search(t).group(0)!r}"})
    for name in by:
        if name not in {i["name"] for i in rep["members"]}:
            out.append({"member": name, "rule": "member missing from report"})
    return out


VERIFIER_SYSTEM = """You are an independent verifier. You see only measured numbers and a list of claims \
written by another agent. For each claim decide whether the numbers support it. Be strict: a claim is \
supported only if every number it states matches the evidence (within the stated uncertainty) and its \
conclusion follows. Reply with JSON only: a list of objects {"member": str, "claim": str, \
"verdict": "agree" | "disagree", "reason": str}."""


def run_verifier(report: dict, actions: dict, provider: str = "baseten") -> list[dict]:
    rep = actions.get("report") or {"members": []}
    claims = [{"member": i["name"], "claim": i["statement"], "status": i["status"]} for i in rep["members"]]
    claims += [{"member": f["name"], "claim": f"[{f['severity']}] {f['finding']}", "status": "flag"}
               for f in actions["flags"]]
    evidence = {"timing": report["timing"],
                "members": {m["name"]: {"peaks": m["peaks"], "ladder": m["ladder"], "tension": m.get("tension"),
                                        "quality": m["quality"]} for m in report["members"]}}
    msg = [{"role": "system", "content": VERIFIER_SYSTEM},
           {"role": "user", "content": "EVIDENCE:\n" + json.dumps(evidence, default=float)[:60000] +
                                       "\n\nCLAIMS:\n" + json.dumps(claims)}]
    r = chat(msg, provider=provider, max_tokens=6000)
    text = r["message"].get("content") or ""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        verdicts = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        verdicts = []
    if not verdicts:
        verdicts = [{"member": c["member"], "claim": c["claim"], "verdict": "disagree",
                     "reason": "verifier returned no parseable verdict"} for c in claims]
    return verdicts


def render_markdown(report: dict, actions: dict, violations: list[dict], verdicts: list[dict],
                    mode: str, approval: dict | None = None) -> str:
    rep = actions.get("report") or {"summary": "(agent did not submit a report)", "members": [], "next_steps": []}
    disputed = {v["member"] for v in verdicts if v.get("verdict") != "agree"} | {v["member"] for v in violations}
    status = "APPROVED" if approval else "AWAITING ENGINEER APPROVAL"
    L = [f"# Resonance inspection report ({mode} mode)", "",
         f"**Status: {status}**" + (f" by {approval['by']} on {approval['at']}" if approval else ""), "",
         f"Video: `{report.get('video', '?')}` · {report['timing']['frames']} frames at "
         f"{report['timing']['fps']:.1f} fps ({report['timing']['source']})", "",
         "## Summary", "", rep["summary"], "", "## Members", "",
         "| Member | Status | f0 (Hz) | Modes | Statement | Verifier |", "|---|---|---|---|---|---|"]
    by = {m["name"]: m for m in report["members"]}
    for item in rep["members"]:
        m = by.get(item["name"], {})
        lad = m.get("ladder", {})
        f0 = f"{lad['f0_hz']:.3f}" if lad.get("ok") else "—"
        modes = ",".join(str(x["n"]) for x in lad.get("modes", [])) if lad.get("ok") else "—"
        ver = "needs engineer" if item["name"] in disputed else "agrees"
        L.append(f"| {item['name']} | {item['status']} | {f0} | {modes} | {item['statement']} | {ver} |")
    if actions["flags"]:
        L += ["", "## Flags", ""] + [f"- **{f['severity']}** · {f['name']}: {f['finding']} "
                                     f"(evidence: {', '.join(f.get('evidence', []))})" for f in actions["flags"]]
    if actions["reshoots"]:
        L += ["", "## Re-shoot requests", ""] + [f"- {r['name']}: {r['reason']}. **Do:** {r['instructions']}"
                                                 for r in actions["reshoots"]]
    if violations:
        L += ["", "## Rule checks (code)", ""] + [f"- {v['member']}: {v['rule']}" for v in violations]
    L += ["", "## Independent verification (Muse Spark Open)", ""] + \
         [f"- {v.get('member')}: **{v.get('verdict')}**: {v.get('reason')}" for v in verdicts]
    L += ["", "## Next steps", ""] + [f"- {s}" for s in rep.get("next_steps", [])]
    L += ["", "---", "Screening measurement from video. It does not certify any structure; "
               "an engineer must review and approve it."]
    return "\n".join(L) + "\n"


def inspect(report: dict, context: str, out_dir: str | Path, mode: str = "bench",
            inspector: str = "baseten", verifier: str = "baseten", log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    actions = run_inspector(report, context, mode=mode, provider=inspector, log=log)
    violations = enforce_rules(report, actions, mode)
    verdicts = run_verifier(report, actions, provider=verifier)
    (out / "agent.json").write_text(json.dumps({"actions": actions, "violations": violations,
                                                "verdicts": verdicts, "mode": mode,
                                                "seconds": time.time() - t0}, indent=2, default=float))
    md = out / "inspection.md"
    md.write_text(render_markdown(report, actions, violations, verdicts, mode))
    return md


def approve(out_dir: str | Path, by: str) -> Path:
    out = Path(out_dir)
    data = json.loads((out / "agent.json").read_text())
    report = json.loads((out / "report.json").read_text())
    approval = {"by": by, "at": time.strftime("%Y-%m-%d %H:%M %Z")}
    data["approval"] = approval
    (out / "agent.json").write_text(json.dumps(data, indent=2, default=float))
    md = out / "inspection.md"
    md.write_text(render_markdown(report, data["actions"], data["violations"], data["verdicts"],
                                  data["mode"], approval))
    return md
