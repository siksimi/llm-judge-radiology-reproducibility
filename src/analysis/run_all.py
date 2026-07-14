"""
run_all.py — run all four analyses, write outputs/tables, and print a summary.
Usage: PYTHONPATH=src/analysis python3 src/analysis/run_all.py
"""
from __future__ import annotations
from pathlib import Path
import reliability, variance, effects, intermodel
from _common import load_cfg

OUT = Path("outputs/tables"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg()
    print("[1/4] reliability (ICC, weighted kappa, CV)")
    r = reliability.run(cfg); r.to_csv(OUT / "reliability.csv", index=False)
    prim = r[r.is_primary] if "is_primary" in r else r
    if len(prim):
        print(prim[[c for c in ["judge_id", "ICC2_1", "ICC2_k", "weighted_kappa", "CV"] if c in prim.columns]].to_string(index=False))

    print("\n[2/4] variance components")
    v = variance.run(cfg); v.to_csv(OUT / "variance_components.csv", index=False)
    print(v.to_string(index=False) if len(v) else "(no result)")

    print("\n[3/4] effects (temperature / prompt / drift)")
    e = effects.run(cfg)
    for name, t in e.items():
        t.to_csv(OUT / f"effects_{name}.csv", index=False)
    print("temperature:\n", e["temperature"].to_string(index=False))
    print("prompt:\n", e["prompt"].to_string(index=False))

    print("\n[4/4] inter-model agreement")
    im = intermodel.run(cfg); im.to_csv(OUT / "intermodel_agreement.csv", index=False)
    print(im.to_string(index=False) if len(im) else "(no result)")

    print(f"\n[done] tables → {OUT}/")


if __name__ == "__main__":
    main()
