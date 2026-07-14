"""
dataset_builder.py
==================
Open-i / Indiana University Chest X-ray reports -> research dataset (reports.jsonl + metadata.csv).

Steps:
  1) Obtain NLM Open-i IU CXR reports (ecgen-radiology XML)
       - --download : auto-download & extract NLMCXR_reports.tgz from NLM (run on a machine with normal internet)
       - --reports-dir : point to an already-extracted XML folder
  2) Extract FINDINGS + IMPRESSION from each XML -> compose report body
  3) Filter usable reports (drop empty/None FINDINGS)
  4) Seed-fixed sampling of n reports
  5) Apply the original:perturbed ratio (default 30:70) — perturbations balanced across 4 families x intensity (perturbation_engine)
  6) Save data/reports.jsonl + data/metadata.csv

Note on network: download the reports on a machine with normal internet access.

NLM download URL (reference):
    reports: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturbation_engine import apply_perturbation, PERTURBERS  # noqa: E402

REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"


# --------------------------------------------------------------------- #
# 1) Obtain
# --------------------------------------------------------------------- #

def download_reports(dest_dir: Path) -> Path:
    """Download NLMCXR_reports.tgz and extract it. Returns: the XML root folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tgz = dest_dir / "NLMCXR_reports.tgz"
    if not tgz.exists():
        print(f"[download] {REPORTS_URL}")
        urllib.request.urlretrieve(REPORTS_URL, tgz)
    print(f"[extract] {tgz}")
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(dest_dir)
    # typically *.xml under ecgen-radiology/
    candidates = list(dest_dir.rglob("*.xml"))
    if not candidates:
        raise RuntimeError("No XML found after extraction.")
    return candidates[0].parent


# --------------------------------------------------------------------- #
# 2) Parse
# --------------------------------------------------------------------- #

def parse_iu_xml(xml_path: Path) -> dict | None:
    """
    Extract sections from an IU ecgen-radiology XML.
    Structure: <Abstract><AbstractText Label="FINDINGS">...</AbstractText> ...
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    sections = {}
    for at in root.iter("AbstractText"):
        label = (at.get("Label") or "").strip().upper()
        text = (at.text or "").strip()
        if label:
            sections[label] = text

    # report identifier
    uid = None
    for tag in ("uId", "IUXRId", "pmcId"):
        el = root.find(f".//{tag}")
        if el is not None and el.get("id"):
            uid = f"{tag}:{el.get('id')}"
            break
    if uid is None:
        uid = xml_path.stem

    findings = sections.get("FINDINGS", "").strip()
    impression = sections.get("IMPRESSION", "").strip()

    return {
        "uid": uid,
        "findings": findings,
        "impression": impression,
        "indication": sections.get("INDICATION", "").strip(),
        "comparison": sections.get("COMPARISON", "").strip(),
    }


def _is_usable(rec: dict) -> bool:
    f = rec["findings"].lower()
    if not f or f in {"none.", "none", "xxxx.", ""}:
        return False
    # drop threadbare reports left with only anonymization placeholders (XXXX) (can be relaxed)
    alpha = f.replace("x", "").replace(".", "").strip()
    return len(alpha) >= 15


def compose_text(rec: dict) -> str:
    parts = []
    if rec["findings"]:
        parts.append(f"FINDINGS: {rec['findings']}")
    if rec["impression"]:
        parts.append(f"IMPRESSION: {rec['impression']}")
    return "  ".join(parts)


# --------------------------------------------------------------------- #
# 3) Perturbation assignment
# --------------------------------------------------------------------- #

def assign_perturbations(n_perturbed: int, seed: int) -> list[tuple[str, int]]:
    """Balance-assign n_perturbed items across 4 perturbation families x intensity (1,2,3)."""
    rng = random.Random(f"{seed}|assign")
    combos = [(pt, inten) for pt in PERTURBERS for inten in (1, 2, 3)]
    out = []
    while len(out) < n_perturbed:
        rng.shuffle(combos)
        out.extend(combos)
    return out[:n_perturbed]


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def build(reports_dir: Path, n: int, ratio: tuple[int, int], seed: int,
          out_jsonl: Path, out_meta: Path):
    xmls = sorted(reports_dir.rglob("*.xml"))
    print(f"[parse] {len(xmls)} XML files")
    parsed = [r for x in xmls if (r := parse_iu_xml(x)) and _is_usable(r)]
    print(f"[filter] usable reports: {len(parsed)}")
    if len(parsed) < n:
        raise RuntimeError(f"Usable reports {len(parsed)} < requested n={n}")

    rng = random.Random(f"{seed}|sample")
    sample = rng.sample(parsed, n)

    n_orig = round(n * ratio[0] / sum(ratio))
    n_pert = n - n_orig
    perturb_plan = assign_perturbations(n_pert, seed)

    rng.shuffle(sample)
    originals, to_perturb = sample[:n_orig], sample[n_orig:]

    rows, meta = [], []
    for rec in originals:
        rid = f"{rec['uid']}_orig"
        rows.append({"id": rid, "source": "open-i", "original_uid": rec["uid"],
                     "perturbation_type": "none", "intensity": 0,
                     "text": compose_text(rec)})
        meta.append({"id": rid, "uid": rec["uid"], "perturbation_type": "none",
                     "intensity": 0, "operations": ""})

    for rec, (pt, inten) in zip(to_perturb, perturb_plan):
        base = compose_text(rec)
        res = apply_perturbation(rec["uid"], base, pt, inten, seed=seed)
        rid = f"{rec['uid']}_{pt}_i{inten}"
        rows.append({"id": rid, "source": "open-i", "original_uid": rec["uid"],
                     "perturbation_type": pt, "intensity": inten,
                     "text": res.perturbed_text})
        meta.append({"id": rid, "uid": rec["uid"], "perturbation_type": pt,
                     "intensity": inten, "operations": "; ".join(res.operations)})

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_meta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "uid", "perturbation_type", "intensity", "operations"])
        w.writeheader(); w.writerows(meta)

    print(f"[done] wrote {len(rows)} reports → {out_jsonl}")
    print(f"       original={n_orig}, perturbed={n_pert} (ratio {ratio[0]}:{ratio[1]})")
    print(f"       metadata → {out_meta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download the reports tgz from NLM")
    ap.add_argument("--reports-dir", type=str, default="data/openi_reports",
                    help="extracted XML folder (or download destination folder)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ratio", type=str, default="30:70", help="original:perturbed")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out-jsonl", type=str, default="data/reports.jsonl")
    ap.add_argument("--out-meta", type=str, default="data/metadata.csv")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    if args.download:
        reports_dir = download_reports(reports_dir)
    ratio = tuple(int(x) for x in args.ratio.split(":"))
    build(reports_dir, args.n, ratio, args.seed,
          Path(args.out_jsonl), Path(args.out_meta))


if __name__ == "__main__":
    main()
