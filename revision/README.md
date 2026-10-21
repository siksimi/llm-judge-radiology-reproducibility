# Revision (R1) materials — DHJ-26-2797

Materials added for the first revision of the manuscript (October 2026). The original study code and data are in the repository root; corrections are listed in `../ERRATA.md`.

| Path | Contents |
|---|---|
| `analysis/` | Revision analysis scripts (Python 3.10 with pandas, statsmodels 0.15.0 and scipy; R 4.3.3 with ordinal 2023.12-4). |
| `analysis/input/` | The 32,000-row per-call score table (identical to `../runs/scores.parquet`), the source-to-input audit table (`report_status.csv`: assigned versus realized perturbation status) and the C1 table used by the ordinal models. |
| `analysis/out/` | Outputs reported in the manuscript and supplement. |
| `campaigns/` | Code, few-shot prompts and per-call score tables of the two additional runs on 8–9 October 2026: C-F (v1, v5 as transmitted, v5 without developer notes, and v5′; 8,000 evaluations) and C-Q (re-run of Qwen 3.5 in C1; 1,000 evaluations). The transmitted v5 block is `../prompts/fewshot_block.txt`. |
| `panel/` | De-identified radiologist ratings: blinded scoring by two independent radiologists (`task1_readerA.csv`, `task1_readerB.csv`), the paired review by two co-author radiologists (`task2_responses.csv`; raters R1 and R2), the sampling manifest and the form-code key. |

Raw model outputs are not included; the per-call score tables allow every reported statistic to be recomputed.

## Reproducing the revision results (from `analysis/`)

```
python reanalysis_phase1.py
python reanalysis_phase2_mixed.py
python r1_12_vc_bootstrap.py input/scores_rebuilt_32000.parquet out/r1_12_vc_bootstrap.csv 2000 20261021
(cd input && mkdir -p out && Rscript ../r1_3_ordinal.R low && Rscript ../r1_3_ordinal.R high)
(cd input && Rscript ../r1_10_clmm.R)
python campaign_analysis.py
python panel_analysis.py
python make_figures_R1.py out <figure_dir> out/r1_12_vc_bootstrap.csv
python r2_5_source_characteristics.py ../.. out/r2_5_source_characteristics.csv
```

Bootstrap seeds are fixed in each script. The ordinal models write `input/out/{low,high}_*.csv` (reported as `out/r1_3_{low,high}_*.csv`). `r2_5_source_characteristics.py` requires the Open-i XML reports in `../data/openi_reports/` (not redistributed).
