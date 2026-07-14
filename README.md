# Reliability and Prompt-Sensitivity of LLMs as Evaluators of Radiology Report Quality

A methodological **reproducibility study** of large language models (LLMs) used as *evaluators* ("LLM-as-judge") of chest-radiograph report quality. We quantify how a judge's quality scores vary under repeated execution, sampling temperature, and semantically equivalent prompt rewordings, and how different judge models agree with one another.

> No patients, no human raters (the judges are LLMs); public/de-identified or synthetic inputs; non-human-subjects / IRB-exempt design.

## Study at a glance

100 chest-radiograph reports (Open-i / Indiana University CXR) are given a quality spectrum via deterministic, rule-based perturbations. Four LLM judges score every report on a five-item 1–5 rubric across three crossed conditions.

| Condition | Manipulation | Target |
|---|---|---|
| **C1** repeat | fixed setting, N = 10 repeats | intra-judge reproducibility (primary) |
| **C2** temperature | T = 0 / 0.3 / 0.7 / 1.0 | effect of temperature on variance |
| **C3** prompt | 5 semantically equivalent variants (v1–v5) | prompt sensitivity & ranking stability |

**Judges:** Claude Opus 4.8 · GPT-5.5 · Gemini 2.5 Pro · **Qwen 3.5 (27B, dense; open-weight, self-hosted)**
**Primary outcome:** ICC(2,1) / ICC(2,k) of `overall_global_quality`.

## Key findings (32,000 evaluations; 100% parsing)

- Intra-judge reliability was excellent for all judges (ICC[2,1] 0.91–1.00), but only the **self-hosted open-weight judge was perfectly reproducible at temperature 0** (ICC = 1.000, CV = 0); commercial APIs varied run-to-run even at fixed settings (Gemini remained non-deterministic at temperature 0).
- **Temperature inflated dispersion, not the mean** (within-report SD rose monotonically with T).
- **Prompt wording** shifted absolute scores modestly (≤ ~0.33 on the 1–5 scale) while **preserving report rankings** (Kendall's W 0.86–0.95).
- **Inter-model agreement was only moderate** (weighted κ 0.67–0.80; exact 41–61%) — judges are not interchangeable.

See `supplementary_reproducibility.md` for full numbers, exact settings, and definitions.

## Repository layout

```
config.yaml                     # all frozen run parameters (judges, conditions, seed)
prompts/                        # system, rubric (+reversed), few-shot, v1–v5
schemas/score.schema.json       # judge output schema (5 items, 1–5)
src/                            # pipeline
  perturbation_engine.py         #   section-aware perturbations
  dataset_builder.py             #   Open-i fetch/parse/perturb -> reports.jsonl
  llm_client.py                  #   provider-agnostic client (retry, cache, logging)
  judge_runner.py                #   orchestrates C1–C3
  response_parser.py             #   JSON + regex-fallback parser
  analysis/                      #   reliability, variance, effects, intermodel, figures
serve/                          # self-hosting Qwen (vLLM) on AMD/Vast + runbooks
tools/merge_qwen_results.py     # offline merge of self-hosted results into scores.parquet
runs/scores.parquet             # aggregated per-call scores (32,000 rows)
```

## Reproduce

```bash
pip install -r requirements.txt

# 1) build the report set (run locally; Open-i download)
python src/dataset_builder.py --download --n 100

# 2) score (set provider API keys; Qwen via a self-hosted OpenAI-compatible endpoint)
#    self-hosting: see serve/README_vast.md or serve/README_amd.md
python -c "import sys;sys.path.insert(0,'src');import judge_runner;judge_runner.run()"

# 3) analysis + figures
PYTHONPATH=src/analysis python src/analysis/run_all.py
PYTHONPATH=src/analysis python src/analysis/make_figures.py
```

Reproducibility details (exact model snapshots, dates, parameters, software versions, prompts, schema, parsing rules) are in **`supplementary_reproducibility.md`**.

## Data & code availability

- **Code:** this repository (MIT License, see `LICENSE`).
- **Aggregated scores:** `runs/scores.parquet` (per-call parsed scores). Raw model outputs are shared subject to provider terms.
- **Source reports:** public Open-i / Indiana University CXR collection (used within its terms). Raw dumps are not redistributed here; regenerate with `dataset_builder.py`.
- An archival release (with data) will be minted as a Zenodo DOI from a tagged GitHub release.

## Citation

If you use this repository, please cite the accompanying paper (citation to be added on acceptance). Authors: Jei Youn Park\*, Suji Lee\*, Youho Myong†, Yongsik Sim† (\*equal contribution; †co-corresponding).

## License

MIT for code (`LICENSE`). Aggregated data are shared within the license scope of the source datasets.
