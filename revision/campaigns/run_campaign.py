"""run_campaign.py v2 — revision-stage new-call campaigns for DHJ-26-2797 (prepared 2026-10-06; NOT yet run).

Campaigns (fresh cache namespace; the June cache is never read):
  CF  few-shot contrast: 4 judges x 100 reports x {v1, v5_as_sent, v5_clean, v5prime} x 5 runs = 8,000 logical calls
      (commercial APIs 6,000 + self-hosted Qwen 2,000)
  CQ  Qwen two-occasion C1: qwen-3.5 x 100 reports x v1 x 10 runs (T=0) = 1,000 calls (vLLM endpoint). CF Qwen-v1 is NOT reused for CQ.

Steps (run from the YH2 repo root, keys in env):
  --plan        freeze manifest (config/prompt/data hashes) and print plan; no calls
  --preflight   1 uncached call per judge; exits 1 on any API error / parse failure; writes preflight receipt
  --go          requires a passing receipt for the SAME manifest; stops on fatal errors; checkpoints every 200 calls
Guards: cache_tag includes the manifest hash, so any change to models/prompts/data creates new keys instead of reusing old answers.
"""
import argparse, hashlib, json, os, random, sys, time, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('YH2_REPO', HERE.parent.parent))   # repository root
sys.path.insert(0, str(REPO / 'src')); sys.dont_write_bytecode = True
import yaml
from llm_client import LLMClient, judges_from_config
from judge_runner import PromptBuilder
from response_parser import parse_scores, REQUIRED_ITEMS

FEWSHOT = {'v5_as_sent': REPO / 'prompts/fewshot_block.txt',   # transmitted June v5 block
           'v5_clean': HERE / 'prompts/fewshot_v5_clean.txt',
           'v5prime': HERE / 'prompts/fewshot_v5prime_FINAL.txt'}   # FINAL exists only after radiologist sign-off
CAMPAIGN = {'CF': dict(tag='CF2026-10', judges=None, prompts=['v1', 'v5_as_sent', 'v5_clean', 'v5prime'], runs=5, temps=[0.0]),
            'CQ': dict(tag='CQ2026-10', judges=['qwen-3.5'], prompts=['v1'], runs=10, temps=[0.0])}
FATAL = ('401', '403', '404', 'invalid_api_key', 'authentication', 'permission', 'not found', 'NOT_FOUND',
         'credits are depleted', 'insufficient_quota', 'billing', 'INVALID_ARGUMENT', 'model_not_found')
MAX_CONSEC_ERR = 5          # consecutive bad calls (API error OR parse failure) per judge -> stop
OUT_ROOT = Path(os.environ.get('CAMPAIGN_OUT_ROOT', str(HERE)))   # tests point this to a temp dir
MAX_TASKS = int(os.environ.get('CAMPAIGN_MAX_TASKS', '0'))          # tests only; 0 = all

class CampaignClient(LLMClient):
    """Same as LLMClient but records the model version RETURNED by Gemini (response.model_version) instead of the configured name."""
    def _call_google(self, judge, system, prompt, params):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'), http_options=types.HttpOptions(timeout=int(self.request_timeout_s * 1000)))
        cfg = dict(system_instruction=system, max_output_tokens=params['max_tokens'], response_mime_type='application/json')
        if 'temperature' in params:
            cfg['temperature'] = params['temperature']; cfg['top_p'] = params['top_p']
        resp = client.models.generate_content(model=judge.model, contents=prompt, config=types.GenerateContentConfig(**cfg))
        returned = getattr(resp, 'model_version', None)
        return (resp.text or ''), (returned or f'{judge.model}|returned_version_missing')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def manifest(spec, judges):
    files = {'config.yaml': REPO / 'config.yaml', 'reports.jsonl': REPO / 'data/reports.jsonl', 'system.txt': REPO / 'prompts/system.txt',
             'rubric_block.txt': REPO / 'prompts/rubric_block.txt', 'v1.txt': REPO / 'prompts/v1.txt', 'v5.txt': REPO / 'prompts/v5.txt',
             'llm_client.py': REPO / 'src/llm_client.py', 'run_campaign.py': Path(__file__)}
    files.update({f'fewshot_{p}': FEWSHOT[p] for p in spec['prompts'] if p in FEWSHOT})
    m = dict(campaign=spec['tag'], judges=[dict(id=j.id, provider=j.provider, model=j.model, base_url=getattr(j, 'base_url', None)) for j in judges],
             prompts=spec['prompts'], runs=spec['runs'], temps=spec['temps'], files={k: sha(v) for k, v in files.items()})
    m['manifest_hash'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
    return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--campaign', required=True, choices=CAMPAIGN)
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ('--plan', '--preflight', '--go'): g.add_argument(f, action='store_true')
    ap.add_argument('--judge', default=None, help='with --go only: run just this judge (same manifest/receipt), so judges can run in parallel processes')
    a = ap.parse_args(); spec = CAMPAIGN[a.campaign]
    cfg = yaml.safe_load((REPO / 'config.yaml').read_text(encoding='utf-8'))
    judges = [j for j in judges_from_config(cfg['judges']) if spec['judges'] is None or j.id in spec['judges']]
    missing = [str(FEWSHOT[p]) for p in spec['prompts'] if p in FEWSHOT and not FEWSHOT[p].exists()]
    if missing: sys.exit(f'[stop] missing prompt file(s): {missing} (v5prime needs radiologist sign-off -> fewshot_v5prime_FINAL.txt)')
    reports = [json.loads(l) for l in (REPO / 'data/reports.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    pb = PromptBuilder(str(REPO / 'prompts')); max_tokens = cfg['api']['max_tokens']
    shots = {p: FEWSHOT[p].read_text(encoding='utf-8') for p in spec['prompts'] if p in FEWSHOT}
    out = OUT_ROOT / f"runs_{spec['tag']}"; out.mkdir(parents=True, exist_ok=True)
    m = manifest(spec, judges); mh = m['manifest_hash']
    (out / f'manifest_{mh}.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    logname = f'calls_{a.judge}.log.jsonl' if a.judge else 'calls.log.jsonl'
    client = CampaignClient(cache_dir=str(out / 'cache'), log_path=str(out / logname), max_retries=cfg['api']['max_retries'],
                            backoff_base_s=cfg['api']['backoff_base_s'], request_timeout_s=cfg['api'].get('request_timeout_s', 120),
                            use_cache=True, min_interval_s=cfg['api'].get('min_interval_s', 0))
    if client.mock: sys.exit('[stop] YH2_MOCK=1 is set — unset it for a real campaign')
    def prompt_for(p, text):
        if p == 'v1': return pb.build('v1', text)
        pb.fewshot = shots[p]; return pb.build('v5', text)
    tasks = []; rng = random.Random(f"{spec['tag']}|order")
    for run in range(spec['runs']):
        block = [(j, r, p, T) for j in judges for r in reports for p in spec['prompts'] for T in spec['temps']]
        rng.shuffle(block); tasks += [(j, r, p, T, run) for j, r, p, T in block]
    n_comm = sum(1 for t in tasks if t[0].provider != 'openai_compatible')
    print(f"[plan] {a.campaign} tag={spec['tag']} manifest={mh} judges={[j.id for j in judges]} prompts={spec['prompts']} runs={spec['runs']} "
          f"-> {len(tasks)} logical calls (commercial API {n_comm}, self-hosted {len(tasks)-n_comm}); retries extra")
    receipt = out / f'preflight_{mh}.json'
    if a.plan:
        for p in spec['prompts']: print(f'  prompt {p}: {len(prompt_for(p, reports[0]["text"]))} chars (+system {len(pb.system)})')
        return
    if a.preflight:
        res_all, ok = [], True
        for j in judges:
            r = client.complete(j, pb.system, prompt_for('v1', reports[0]['text']), temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                                cache_tag=f"{spec['tag']}|{mh}|PREFLIGHT|{dt.datetime.now().isoformat()}")
            pr = parse_scores(r.raw_text or '') if not r.error else None
            good = (r.error is None) and bool(pr and pr.ok); ok &= good
            res_all.append(dict(judge=j.id, ok=good, model_version=r.model_version, sent_params=r.params, error=r.error, latency_s=r.latency_s))
            print(f"  {j.id}: ok={good} model_version={r.model_version} sent={r.params} error={r.error}")
        receipt.write_text(json.dumps(dict(manifest_hash=mh, time=dt.datetime.now().isoformat(), all_ok=ok, results=res_all), indent=2), encoding='utf-8')
        sys.exit(0 if ok else 1)
    # --go
    if not receipt.exists() or not json.loads(receipt.read_text(encoding='utf-8')).get('all_ok'):
        sys.exit(f'[stop] no passing preflight receipt for manifest {mh} — run --preflight first (and re-run it after any config/prompt change)')
    if a.judge:
        if a.judge not in [j.id for j in judges]: sys.exit(f'[stop] unknown judge {a.judge}')
        tasks = [t for t in tasks if t[0].id == a.judge]
        print(f'[go] judge filter {a.judge}: {len(tasks)} calls')
    if MAX_TASKS: tasks = tasks[:MAX_TASKS]
    import pandas as pd
    rows, consec, done = [], {}, 0
    def save(final=False):
        suffix = f'_{a.judge}' if a.judge else ''
        pd.DataFrame(rows).to_parquet(out / f"scores_{mh}{suffix}{'' if final else '_checkpoint'}.parquet", index=False)
    for j, r, p, T, run in tasks:
        res = client.complete(j, pb.system, prompt_for(p, r['text']), temperature=T, top_p=1.0, max_tokens=max_tokens,
                              cache_tag=f"{spec['tag']}|{mh}|{p}|T{T}|run{run}|{r['id']}")
        pr = parse_scores(res.raw_text or '') if not res.error else None
        row = dict(campaign=spec['tag'], manifest=mh, judge_id=j.id, report_id=r['id'], prompt=p, requested_temperature=T, run=run,
                   sent_params=json.dumps(res.params), model_version=res.model_version, access_date=res.access_date, cache_hit=res.cache_hit,
                   latency_s=res.latency_s, api_error=res.error, parse_ok=bool(pr and pr.ok), parse_reason=(pr.reason if pr else 'api_error'))
        for it in REQUIRED_ITEMS: row[it] = pr.scores.get(it) if pr and pr.ok else None
        rows.append(row); done += 1
        bad = bool(res.error) or not row['parse_ok']
        if res.error and any(tok.lower() in res.error.lower() for tok in FATAL):
            save(); print(f'[STOP] {j.id}: fatal API error -> {res.error[:200]} (progress saved; fix, re-run --preflight, then --go to resume)'); sys.exit(3)
        consec[j.id] = consec.get(j.id, 0) + 1 if bad else 0
        if consec[j.id] >= MAX_CONSEC_ERR:
            why = res.error[:200] if res.error else f'parse failure ({row["parse_reason"]})'
            save(); print(f'[STOP] {j.id}: {consec[j.id]} consecutive bad calls (API error or parse failure) -> {why} (progress saved)'); sys.exit(4)
        if done % 200 == 0: save(); print(f'  {done}/{len(tasks)}', flush=True)
    save(final=True)
    n_err = sum(r['api_error'] is not None for r in rows); n_ok = sum(r['parse_ok'] for r in rows)
    n_parse_fail = sum((r['api_error'] is None) and not r['parse_ok'] for r in rows)
    print(f'[done] rows={len(rows)} valid={n_ok} api_error={n_err} parse_fail={n_parse_fail}')
    sys.exit(0 if n_ok == len(rows) else 2)   # any API error or parse failure -> non-zero (inspect before analysis)

if __name__ == '__main__':
    main()
