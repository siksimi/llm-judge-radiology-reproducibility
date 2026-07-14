"""
judge_runner.py — experiment orchestrator (reports x judges x conditions C1-C4).

Flow: load config -> load reports.jsonl -> build tasks -> inject prompt ->
      llm_client.complete(cache_tag) -> response_parser -> write runs/scores.parquet.
Prompt injection: v2 -> reversed rubric, v5 -> add few-shot.

Interrupt safety:
  - Ctrl+C once -> "stop requested": finish the in-flight call, exit the loop,
    save results so far to scores.parquet, and exit cleanly.
  - Ctrl+C twice -> immediate force quit (no save).
  - Per-call timeout (config api.request_timeout_s) prevents network hangs.
  - Successful calls are cached immediately, so re-running resumes with zero duplicate calls.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import yaml
import pandas as pd

from llm_client import LLMClient, judges_from_config, JudgeSpec
from response_parser import parse_scores, REQUIRED_ITEMS


class PromptBuilder:
    def __init__(self, prompts_dir="prompts"):
        d = Path(prompts_dir)
        self.system = (d / "system.txt").read_text(encoding="utf-8")
        self.rubric = (d / "rubric_block.txt").read_text(encoding="utf-8")
        self.rubric_rev = (d / "rubric_block_reversed.txt").read_text(encoding="utf-8")
        self.fewshot = (d / "fewshot_block.txt").read_text(encoding="utf-8")
        self._templates = {v: (d / f"{v}.txt").read_text(encoding="utf-8")
                           for v in ["v1", "v2", "v3", "v4", "v5"]}

    def build(self, variant, report_text):
        tpl = self._templates[variant]
        tpl = "\n".join(l for l in tpl.splitlines() if not l.startswith("#"))
        rubric = self.rubric_rev if variant == "v2" else self.rubric
        out = tpl.replace("{RUBRIC_BLOCK}", rubric)
        out = out.replace("{FEWSHOT_BLOCK}", self.fewshot if variant == "v5" else "")
        out = out.replace("{REPORT}", report_text)
        return out.strip()


def _mk(judge, rep, condition, variant, T, top_p, run):
    return {"judge_id": judge.id, "family": judge.family,
            "report_id": rep["id"], "perturbation_type": rep.get("perturbation_type", "none"),
            "intensity": rep.get("intensity", 0), "condition": condition, "variant": variant,
            "temperature": T, "top_p": top_p, "run": run, "_report_text": rep["text"]}


def build_tasks(cfg, reports, judges):
    tasks = []
    cond = cfg["conditions"]
    for judge in judges:
        c1 = cond["C1_repeat"]
        if c1.get("enabled"):
            for rep in reports:
                for run in range(c1["n_repeats"]):
                    tasks.append(_mk(judge, rep, "C1", "v1", c1["temperature"], c1["top_p"], run))
        c2 = cond["C2_temperature"]
        if c2.get("enabled"):
            skip = c2.get("skip_if_not_temperature_capable") and (judge.temperature_capable is False)
            if not skip:
                for T in c2["grid"]:
                    for rep in reports:
                        for run in range(c2["n_repeats"]):
                            tasks.append(_mk(judge, rep, "C2", "v1", T, c2["top_p"], run))
        c3 = cond["C3_prompt"]
        if c3.get("enabled"):
            for variant in c3["prompt_variants"]:
                for rep in reports:
                    for run in range(c3["n_repeats"]):
                        tasks.append(_mk(judge, rep, "C3", variant, c3["temperature"], 1.0, run))
    return tasks


def _save(rows, cfg):
    df = pd.DataFrame(rows)
    out_path = Path(cfg["paths"]["runs_dir"]) / "scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
    return df


def run(config_path="config.yaml", limit=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    reports = [json.loads(l) for l in
               Path(cfg["paths"]["reports_jsonl"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    judges = judges_from_config(cfg["judges"])
    # Selective run: if YH2_ONLY_JUDGES="qwen-3.5" (comma-separated), run only those judges.
    # Used to run only Qwen on a self-host instance. The cache key is identical, so closed models
    # run elsewhere with the same config merge safely into scores.parquet.
    _only = os.environ.get("YH2_ONLY_JUDGES", "").strip()
    if _only:
        keep = {s.strip() for s in _only.split(",") if s.strip()}
        judges = [j for j in judges if j.id in keep]
        print(f"[run] YH2_ONLY_JUDGES={sorted(keep)} → judges={[j.id for j in judges]}", flush=True)
        if not judges:
            raise SystemExit(f"[run] no judge in config matches YH2_ONLY_JUDGES={keep}")
    pb = PromptBuilder()
    client = LLMClient(cache_dir=str(Path(cfg["paths"]["runs_dir"]) / "cache"),
                       log_path=str(Path(cfg["paths"]["runs_dir"]) / "calls.log.jsonl"),
                       max_retries=cfg["api"]["max_retries"],
                       backoff_base_s=cfg["api"]["backoff_base_s"],
                       request_timeout_s=cfg["api"].get("request_timeout_s", 120),
                       use_cache=cfg["api"]["cache"],
                       min_interval_s=cfg["api"].get("min_interval_s", 0))
    max_tokens = cfg["api"]["max_tokens"]
    on_fail = cfg["output"]["on_parse_failure"]
    judge_by_id = {j.id: j for j in judges}

    tasks = build_tasks(cfg, reports, judges)
    if limit:
        tasks = tasks[:limit]

    # ---- Ctrl+C graceful stop ----
    stop = {"req": False, "force": False}
    def _handler(signum, frame):
        if stop["req"]:
            stop["force"] = True
            print("\n[run] Ctrl+C twice -> immediate force quit (no save)", flush=True)
            raise KeyboardInterrupt
        stop["req"] = True
        print("\n[run] stop requested — finishing the in-flight call, then saving and exiting. (Ctrl+C again = force)", flush=True)
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass

    rows = []
    n_total = len(tasks); n_done = n_cache = n_err = n_skip = 0
    sent_new = {}; capped = set()
    if client.mock:
        print("[run] " + "="*54 + "\n[run] !!!  MOCK MODE (YH2_MOCK=1) — not a real API, fake scores!\n"
              "[run]      to run for real, clear YH2_MOCK then re-run\n[run] " + "="*54, flush=True)
    print(f"[run] tasks={n_total} (judges={len(judges)})  — Ctrl+C to stop safely", flush=True)
    interrupted = False
    try:
        for t in tasks:
            if stop["req"]:
                interrupted = True
                print(f"[run] stopped by request: processed {n_done}/{n_total}", flush=True)
                break
            jid = t["judge_id"]; jspec = judge_by_id[jid]
            prompt = pb.build(t["variant"], t.pop("_report_text"))
            cache_tag = f"{t['condition']}|{t['variant']}|T{t['temperature']}|run{t['run']}|{t['report_id']}"
            # daily cap: once new (uncached) calls reach the cap, skip this judge's remaining new calls
            if jspec.daily_cap and sent_new.get(jid, 0) >= jspec.daily_cap:
                if not client.cached_exists(jspec, pb.system, prompt,
                                            temperature=t["temperature"], top_p=t["top_p"],
                                            max_tokens=max_tokens, cache_tag=cache_tag):
                    n_skip += 1
                    if jid not in capped:
                        capped.add(jid)
                        print(f"  [{jid}] daily cap ({jspec.daily_cap}) reached -> deferring remaining new calls to a later run", flush=True)
                    continue
            res = client.complete(jspec, pb.system, prompt,
                                  temperature=t["temperature"], top_p=t["top_p"],
                                  max_tokens=max_tokens, cache_tag=cache_tag)
            if not res.cache_hit and not res.error:
                sent_new[jid] = sent_new.get(jid, 0) + 1
            n_done += 1
            if res.cache_hit: n_cache += 1
            if res.error: n_err += 1
            if n_done % 10 == 0 or n_done == n_total:
                print(f"  [{n_done}/{n_total}] cache_hit={n_cache} api_error={n_err}", flush=True)
            row = dict(t)
            row.update({"model_version": res.model_version, "access_date": res.access_date,
                        "cache_hit": res.cache_hit, "api_error": res.error})
            if res.error:
                row["parse_ok"] = False; row["parse_reason"] = "api_error"
            else:
                pr = parse_scores(res.raw_text)
                row["parse_ok"] = pr.ok; row["parse_reason"] = pr.reason
                for item in REQUIRED_ITEMS:
                    row[item] = pr.scores.get(item) if pr.ok else None
            if not (not row["parse_ok"] and on_fail == "drop"):
                rows.append(row)
    except KeyboardInterrupt:
        if stop["force"]:
            print("[run] force quit: this session's results not saved (cache preserved; resume on re-run).", flush=True)
            raise
        interrupted = True

    df = _save(rows, cfg)
    tag = "saved (interrupted)" if interrupted else "done"
    print(f"[run] {tag}: {n_done}/{n_total} calls, cache_hit={n_cache}, api_error={n_err}, "
          f"daily_cap_skip={n_skip}, parse_ok={round(df['parse_ok'].mean(),3) if len(df) else 'NA'}", flush=True)
    if n_skip:
        print(f"[run] {n_skip} calls deferred by the daily cap — re-run the same command tomorrow (or after the quota resets) to continue.", flush=True)
    if interrupted:
        print("[run] results so far saved to scores.parquet. Re-run the same command to process only what remains.", flush=True)
    elif n_err:
        print(f"[run] {n_err} calls failed (credits/network/etc.). Re-run after topping up to retry only the failures.", flush=True)
    return df


if __name__ == "__main__":
    os.environ.setdefault("YH2_MOCK", "1")
    df = run()
    print("rows:", len(df), "parse_ok:", round(df["parse_ok"].mean(), 3) if len(df) else "NA")
