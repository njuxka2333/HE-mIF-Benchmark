"""Leakage-free Immunocto partition.

Immunocto is derived from the Orion CRC cohort (CRC01-CRC40). MIPHEI-ViT is
trained on that same cohort, and its released split is pinned in the authors'
repository (``scripts/train_val_test_split_orion.py``):

    TEST  19510_C11_..._151039        18459_LSP10364_..._092347
    VAL   19510_C19_...               19510_C30_...
    (all 36 remaining Orion slides are training data)

The Orion data release maps those scan ids onto specimen ids
(https://github.com/labsyspharm/orion-crc, datarelease-README.md): the
``18459_LSP*`` prefix covers C1-C10 and ``19510_C##`` covers C11-C40, giving

    18459_LSP10364 -> CRC02      19510_C11 -> CRC11
    19510_C19      -> CRC19      19510_C30 -> CRC30

so **CRC02, CRC11, CRC19 and CRC30 are the only Orion slides MIPHEI-ViT never
trained on**. The benchmark's original test split (CRC33-CRC40) lay entirely
inside its training set, which made the MIPHEI-ViT-vs-UNI2 comparison on
Immunocto uninterpretable. This assignment **is** the official Immunocto split
(``patient_split.csv``): those four slides are test, CRC37–40 are
valid, and the remaining 32 are train. Surplus-``other`` cells are capped per
split to that split's ``cd4`` count (seed 42) and frozen in
``other_excluded.csv.gz``.

Two residual caveats, to be reported rather than resolved:

* CRC11 and CRC02 were MIPHEI-ViT's *test* slides (untouched); CRC19 and CRC30
  were its *validation* slides, so no gradients reached them but they informed
  checkpoint selection.
* MIPHEI-ViT still trained on 36 of the 40 slides in this cohort, so it retains
  in-domain adaptation to Orion tissue and to the Orion scanner even on unseen
  slides. That is ordinary domain advantage, not leakage.

No other benchmarked model uses Orion: ROSIE's 18 training studies (published
manifest) contain no colorectal cohort and no Orion data, HEMIT trains on
ImmunoAIzer, HistoPlexer on TuPro melanoma, GigaTIME on an in-house cohort, and
UNI2 is self-supervised with no Immunocto labels.
"""

from __future__ import annotations

# MIPHEI-ViT's held-out Orion slides, by Immunocto specimen id.
MIPHEIVIT_TEST_SLIDES = ("CRC02", "CRC11")        # never seen
MIPHEIVIT_VALID_SLIDES = ("CRC19", "CRC30")       # selection only, no gradients
LEAKFREE_TEST = tuple(sorted(MIPHEIVIT_TEST_SLIDES + MIPHEIVIT_VALID_SLIDES))

# Validation slides for the benchmark's own probe, drawn from the 36 remaining.
# Fixed here so every arm trains and selects on identical data.
LEAKFREE_VALID = ("CRC37", "CRC38", "CRC39", "CRC40")

ALL_SLIDES = tuple(f"CRC{i:02d}" for i in range(1, 41))
LEAKFREE_TRAIN = tuple(s for s in ALL_SLIDES
                       if s not in LEAKFREE_TEST and s not in LEAKFREE_VALID)

# Public names: this is the only Immunocto split.
TEST, VALID, TRAIN = LEAKFREE_TEST, LEAKFREE_VALID, LEAKFREE_TRAIN


def split_of(case_id: str) -> str:
    """Return 'train' | 'valid' | 'test' for an Immunocto specimen id."""
    cid = str(case_id).strip().upper()
    if cid in LEAKFREE_TEST:
        return "test"
    if cid in LEAKFREE_VALID:
        return "valid"
    if cid in LEAKFREE_TRAIN:
        return "train"
    raise KeyError(f"unknown Immunocto case id: {case_id!r}")


def assignment() -> dict[str, tuple[str, ...]]:
    return {"train": LEAKFREE_TRAIN, "valid": LEAKFREE_VALID, "test": LEAKFREE_TEST}


def describe() -> str:
    a = assignment()
    return (f"Immunocto leakage-free partition: "
            f"train={len(a['train'])} valid={len(a['valid'])} test={len(a['test'])} "
            f"(test = {', '.join(a['test'])})")
