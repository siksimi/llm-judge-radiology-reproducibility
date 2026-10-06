# Errata

## 2026-10-06

An audit of this repository against the cached model responses from the June 2026 runs identified the following errors. All corrections were made without any new model calls.

### 1. `runs/scores.parquet` was not the study score table

From the initial public release (2026-07-15, commit `1c890c0`) until this correction, `runs/scores.parquet` contained a 40-row output of a software smoke test run in mock mode (Qwen 3.5 only; `model_version` ending in `@MOCK`; synthetic scores). It is not study data and was committed in error in place of the study score table; the README nevertheless described it as the 32,000-row per-call score table.

The file has been replaced with the 32,000-row per-call score table, regenerated offline from the cached June 2026 model responses using the repository's prompt builder, cache keys and parser. Running `src/analysis/run_all.py` on the corrected file reproduces the reliability, temperature, prompt-sensitivity, drift, variance-component and inter-model-agreement tables of the study (verified on 2026-10-06).

### 2. `prompts/fewshot_block.txt` differed from the text transmitted to the judges

The few-shot block transmitted to all four judges for prompt variant v5 in the June 2026 runs ended with two comment lines (an internal note in Korean). These lines were removed from the file on 2026-07-14, after the runs and before the public release, so the released file did not match the transmitted prompt and could not reproduce the v5 calls. The file now contains the transmitted text exactly; this was verified by matching all 4,000 v5 cache keys.

The prompt builder inserts the few-shot block verbatim (stripping of `#` lines applies only to the prompt templates), so these lines were part of every v5 prompt. English translation of the two lines:

> TODO (clinical): the reference ("gold") scores of the two examples above are to be finalized by consensus of the corresponding authors.
> Scoring should be consistent with the anchors so that the few-shot examples do not become a source of bias.

### 3. Parsing rate was not 100%

The README reported "100% parsing". Five of the 32,000 responses (Qwen 3.5, condition C2, temperature 1.0) contained a misspelled required key (e.g., `clarify_structure`, `dialogic_accuracy`) and failed schema validation (`parse_ok = False`); their scores are missing in all analyses. The valid-score rate was 31,995/32,000 (99.98%). All 22,000 responses from the three commercial judges were schema-compliant as returned; no score was recovered by the regular-expression fallback.

The `max_parse_retries` value in `config.yaml` is not used by the code: calls were retried only on API or transport errors, and no model was re-queried because of a malformed response.
