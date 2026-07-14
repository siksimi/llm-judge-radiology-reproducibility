"""
perturbation_engine.py  (section-aware v2)
==========================================
Rule-based radiology-report perturbation engine — section-aware (preserves FINDINGS / IMPRESSION labels).

v2 improvements:
  - Reports are parsed into sections (label, body) so perturbations never touch labels.
  - Perturbations are applied to the appropriate section:
      finding_omission     -> delete sentence(s) from the FINDINGS body (keep >=1 sentence)
      hallucinated_finding -> insert hallucinated sentence(s) inside the FINDINGS body (after the label)
      laterality_measure   -> alter laterality/measurements across section bodies (labels preserved)
      readability          -> shuffle sentence order in FINDINGS + insert filler (section boundaries kept)
  - Reports without labels are handled as a single unlabeled section (backward compatible).

Public API (stable): apply_perturbation(original_id, text, ptype, intensity, seed) -> PerturbationResult
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

# ------------------------------------------------------------------ #
# Lexicons / rule tables (to be finalized after clinical review — current draft)
# ------------------------------------------------------------------ #

HALLUCINATION_POOL = [
    "There is a small left pleural effusion.",
    "Mild cardiomegaly is noted.",
    "A subtle nodular opacity is seen in the right upper lobe.",
    "Patchy airspace opacity in the left lower lobe suggests pneumonia.",
    "Increased interstitial markings consistent with mild pulmonary edema.",
    "A calcified granuloma is present in the right lung.",
]

LATERALITY_SWAP = {
    "left": "right", "right": "left",
    "Left": "Right", "Right": "Left",
    "LEFT": "RIGHT", "RIGHT": "LEFT",
}

VERBOSE_FILLERS = [
    "It should be noted that, generally speaking,",
    "As can be appreciated upon careful review,",
    "For the sake of completeness, it is worth mentioning that",
]

MEASURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(cm|mm)\b", flags=re.IGNORECASE)

# Section label: 2+ uppercase letters (spaces/slashes allowed) + colon
SECTION_LABEL_RE = re.compile(r"([A-Z][A-Z /]{1,30}):\s*")

PRIMARY_SECTION = "FINDINGS"


# ------------------------------------------------------------------ #
# Section parsing / reassembly
# ------------------------------------------------------------------ #

@dataclass
class Section:
    label: Optional[str]
    body: str


def parse_sections(text: str) -> list:
    matches = list(SECTION_LABEL_RE.finditer(text))
    if not matches:
        return [Section(None, text.strip())]
    sections = []
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append(Section(None, pre))
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(Section(label, text[start:end].strip()))
    return sections


def reassemble(sections: list) -> str:
    parts = []
    for s in sections:
        if not s.body:
            continue
        parts.append(f"{s.label}: {s.body}" if s.label else s.body)
    return "  ".join(parts)


def _find_target_idx(sections: list, prefer: str = PRIMARY_SECTION) -> int:
    for i, s in enumerate(sections):
        if s.label and s.label.upper() == prefer:
            return i
    for i, s in enumerate(sections):
        if s.body.strip():
            return i
    return 0


def _sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class PerturbationResult:
    original_id: str
    perturbed_text: str
    perturbation_type: str
    intensity: int
    operations: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ #
# Perturbation functions (mutate the section list in place and return ops)
# ------------------------------------------------------------------ #

def perturb_finding_omission(secs, intensity, rng):
    idx = _find_target_idx(secs)
    sents = _sentences(secs[idx].body)
    if len(sents) <= 1:
        return ["omission_skipped_too_short"]
    k = min(intensity, len(sents) - 1)
    drop = sorted(rng.sample(range(len(sents)), k))
    secs[idx].body = " ".join(s for i, s in enumerate(sents) if i not in drop)
    return [f"omit[{secs[idx].label}]_idx={drop}"]


def perturb_hallucinated_finding(secs, intensity, rng):
    idx = _find_target_idx(secs)
    sents = _sentences(secs[idx].body)
    ops = []
    for _ in range(intensity):
        inj = rng.choice(HALLUCINATION_POOL)
        pos = rng.randint(0, len(sents))
        sents.insert(pos, inj)
        ops.append(f"inject[{secs[idx].label}]@{pos}:{inj!r}")
    secs[idx].body = " ".join(sents)
    return ops


def perturb_laterality_measure(secs, intensity, rng):
    ops = []
    budget = {"lat": intensity, "meas": intensity}

    for s in secs:
        if budget["lat"] <= 0:
            break
        tokens = s.body.split()
        positions = [i for i, t in enumerate(tokens) if t.strip(".,;:") in LATERALITY_SWAP]
        rng.shuffle(positions)
        for i in positions:
            if budget["lat"] <= 0:
                break
            raw = tokens[i]; core = raw.strip(".,;:")
            tokens[i] = raw.replace(core, LATERALITY_SWAP[core])
            ops.append(f"laterality[{s.label}]:{core}->{LATERALITY_SWAP[core]}")
            budget["lat"] -= 1
        s.body = " ".join(tokens)

    for s in secs:
        if budget["meas"] <= 0:
            break
        def _mut(m):
            if budget["meas"] <= 0:
                return m.group(0)
            budget["meas"] -= 1
            val = float(m.group(1)); factor = rng.choice([0.5, 1.5, 2.0])
            newval = round(val * factor, 1)
            ops.append(f"measure[{s.label}]:{m.group(0)}->{newval}{m.group(2)}")
            return f"{newval} {m.group(2)}"
        s.body = MEASURE_RE.sub(_mut, s.body)

    if not ops:
        ops = ["laterality_measure_no_target_found"]
    return ops


def perturb_readability(secs, intensity, rng):
    idx = _find_target_idx(secs)
    sents = _sentences(secs[idx].body)
    ops = []
    if len(sents) > 1:
        order = list(range(len(sents)))
        rng.shuffle(order)
        sents = [sents[i] for i in order]
        ops.append(f"shuffle[{secs[idx].label}]={order}")
    for _ in range(intensity):
        if not sents:
            break
        j = rng.randrange(len(sents))
        filler = rng.choice(VERBOSE_FILLERS)
        sents[j] = filler + " " + sents[j][0].lower() + sents[j][1:]
        ops.append(f"verbose[{secs[idx].label}]@{j}:{filler!r}")
    secs[idx].body = " ".join(sents)
    return ops


PERTURBERS = {
    "finding_omission": perturb_finding_omission,
    "hallucinated_finding": perturb_hallucinated_finding,
    "laterality_measure": perturb_laterality_measure,
    "readability": perturb_readability,
}


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def apply_perturbation(original_id, text, ptype, intensity, seed):
    if ptype == "none" or intensity == 0:
        return PerturbationResult(original_id, text, "none", 0, ["original_unchanged"])
    if ptype not in PERTURBERS:
        raise ValueError(f"Unknown perturbation type: {ptype}")
    rng = random.Random(f"{seed}|{original_id}|{ptype}|{intensity}")
    secs = parse_sections(text)
    ops = PERTURBERS[ptype](secs, intensity, rng)
    return PerturbationResult(original_id, reassemble(secs), ptype, intensity, ops)


if __name__ == "__main__":
    demo = ("FINDINGS: The heart size is normal. The lungs are clear. "
            "There is a 3.2 cm nodule in the left lower lobe. No pleural effusion.  "
            "IMPRESSION: Solitary pulmonary nodule in the left lower lobe.")
    print("ORIGINAL:\n", demo, "\n")
    print("SECTIONS:", [(s.label, s.body[:40]) for s in parse_sections(demo)], "\n")
    for t in PERTURBERS:
        for inten in (1, 2):
            r = apply_perturbation("CXR1", demo, t, inten, seed=20260622)
            print(f"[{t} | intensity={inten}]")
            print(" ", r.perturbed_text)
            print("  ops:", r.operations, "\n")
