"""Common loader / utilities shared by the analysis modules."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml

ITEMS = ["finding_completeness", "diagnostic_accuracy", "clarity_structure",
         "clinical_appropriateness", "overall_global_quality"]


def load_cfg(path="config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_scores(cfg: dict) -> pd.DataFrame:
    runs = Path(cfg["paths"]["runs_dir"])
    pq, csv = runs / "scores.parquet", runs / "scores.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("scores.parquet/csv not found — run judge_runner first")
    # keep valid (parsed) scores only
    df = df[df["parse_ok"] == True].copy()  # noqa: E712
    return df


def primary_item(cfg: dict) -> str:
    return cfg["rubric"].get("primary_outcome", "overall_global_quality")
