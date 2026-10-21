# Reproducibility of LLMs as Judges of Radiology Report Quality

A methodological **reproducibility study** of large language models (LLMs) used as *evaluators* ("LLM-as-judge") of chest-radiograph report quality. We quantify how a judge's quality scores vary under repeated execution, sampling temperature, and single-feature prompt changes; how different judge models agree with one another; and, in an exploratory subset added at revision, how judge scores relate to radiologist scores.

> Inputs are public, de-identified reports (Open-i / Indiana University CXR) with rule-based text modifications. No patient data were collected. In the revision, four radiologists (two co-authors, two independent) rated a 48-report subset as expert raters; their de-identified ratings are in `revision/`.

## Study at a glance

100 chest-radiograph reports are given a controlled score spectrum via deterministic, rule-based text perturbations (planned 30 unmodified : 70 modified; realized 42 textually unchanged : 58 changed, because 12 of 17 reports assigned to the laterality/measurement family contained no eligible target — see `revision/` and ERRATA). Four LLM judges score every report on a five-item 1–5 rubric across three conditions.

| Condition | Manipulation | Target |
|---|---|---|
| **C1** repeat | fixed setting, 10 repeats | intra-judge reproducibility (primary) |
| **C2** temperature | T = 0 / 0.3 / 0.7 / 1.0 (Gemini 2.5 Pro, Qwen 3.5 only) | effect of temperature on dispersion |
| **C3** prompt | 5 variants, each changing one feature of v1 (item order, register, scale labels, few-shot examples) | prompt sensitivity & ranking stability |

**Judges:** Claude Opus 4.8 · GPT-5.5 · Gemini 2.5 Pro · **Qwen 3.5 (27B, dense; open-weight, self-hosted, BF16)**
**Primary outcome:** ICC(2,1) / ICC(2,k) of `overall_global_quality`.

## Key findings (32,000 evaluations; 31,995 valid scores, 99.98%)

- Intra-judge reliability of overall quality was high in the constructed set (ICC[2,1] 0.907–1.000) but lower in the 30 unmodified reports (0.789–1.000) and for finding completeness (0.791 and 0.829 in two judges).
- The self-hosted judge returned identical scores across runs within a session (ICC = 1.000); a re-run four months later matched for 97/100 reports. The commercial judges varied run-to-run at fixed settings, including Gemini 2.5 Pro at temperature 0. Model identity and access mode are confounded, so this does not isolate the serving pathway.
- In the two temperature-enabled judges, temperature increased within-report dispersion with little change in mean score.
- Prompt variants shifted prompt-level means by 0.20–0.33 points and individual report scores by more (mean within-report range 0.29–0.71), while report rankings were largely preserved (tie-corrected Kendall's W 0.92–0.98).
- Judges were not interchangeable (pairwise weighted κ 0.67–0.80; exact agreement 41–61%), and all judges scored lower than the radiologists (mean difference −0.30 to −1.00).

Full numbers, exact model snapshots/dates/parameters, and statistical definitions are in the paper's Supplementary Material.

## Repository layout

```
config.yaml                     # all frozen run parameters (judges, conditions, seed)
requirements.txt
prompts/                        # system, rubric (+reversed), few-shot (exactly as transmitted; see ERRATA.md), v1–v5
schemas/score.schema.json       # judge output schema (5 items, 1–5)
src/
  perturbation_engine.py         # section-aware perturbations
  dataset_builder.py             # Open-i fetch/parse/perturb -> reports.jsonl
  llm_client.py                  # provider-agnostic client (retry, cache, logging)
  judge_runner.py                # orchestrates C1–C3
  response_parser.py             # JSON + regex-fallback parser
  analysis/                      # _common, reliability, variance, effects, intermodel, run_all
data/reports.jsonl              # 100 reports (id, source, assigned perturbation, text)
data/metadata.csv               # per-report perturbation family/intensity/operations
runs/scores.parquet             # per-call parsed scores (32,000 rows; parse_ok = False for 5)
revision/                       # revision analyses, additional runs (Oct 2026), v5' prompt,
                                # audit of assigned vs realized perturbation status, radiologist ratings
```

## Reproduce

```bash
pip install -r requirements.txt

# 1) build the report set (run on a machine with internet; Open-i download)
python src/dataset_builder.py --download --n 100

# 2) score all four judges
#    - closed judges: set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
#    - Qwen 3.5: serve the weights with any vLLM (>=0.17) OpenAI-compatible endpoint,
#      then point the client at it via QWEN_BASE_URL (default http://localhost:8000/v1)
python -c "import sys;sys.path.insert(0,'src');import judge_runner;judge_runner.run()"

# 3) analysis -> outputs/tables/*.csv
PYTHONPATH=src/analysis python src/analysis/run_all.py
```

The provided `runs/scores.parquet` reproduces every table of the original analysis without re-running the models; the revision analyses are run from `revision/` (see `revision/README.md`).

## Corrections

Corrections to this repository and to the originally reported figures are documented in [`ERRATA.md`](ERRATA.md): score table replaced (a test file had been deposited), few-shot prompt file restored to the transmitted text, valid-score rate corrected to 99.98%, realized perturbation allocation (42/58) documented, and Kendall's W recomputed with tie correction.

## Data & code availability

- **Code:** this repository (MIT License, see `LICENSE`).
- **Scores:** `runs/scores.parquet` (per-call parsed scores) and the per-call tables of the additional October 2026 runs in `revision/`. Raw model outputs are not deposited; the per-call parsed score tables allow all reported statistics to be recomputed.
- **Radiologist ratings:** de-identified, in `revision/`.
- **Source reports:** public Open-i / Indiana University CXR collection (used within its terms). Raw dumps are not redistributed here; regenerate with `dataset_builder.py`.
- An archival release will be minted as a Zenodo DOI from a tagged GitHub release on acceptance.

## Citation

If you use this repository, please cite the accompanying paper (citation to be added on acceptance). Authors: Jei Youn Park\*, Suji Lee\*, Youho Myong†, Yongsik Sim† (\*equal contribution; †co-corresponding).

## License

MIT for code (`LICENSE`). Aggregated data are shared within the license scope of the source datasets.
