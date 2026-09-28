# Results Summary

All numbers below are single-run results (fixed seed = 42 for the main
BiomedCLIP run; other runs used ad-hoc seeds during iterative debugging —
see [Limitations](../README.md#limitations) in the main README). Treat
differences smaller than ~4–5 points as **not** reliably significant until
multi-seed repetition is done.

> **Known issue:** in the runs where ViT blocks were unfrozen (sections 3–4),
> the global `model` object may have been modified in place by earlier runs in
> the same session (`LinearProbe` keeps a reference, not a copy), so those
> results may not start from the original pretrained weights. Fully frozen
> runs (sections 1–2) are unaffected. Sections 3–4 should be re-run with the
> weights restored before each run; see "Limitations" in the main README.

## 1. Zero-shot classification (no training, 4-way classification, random baseline = 25%)

| Prompt template | BiomedCLIP | PubMedCLIP |
|---|---|---|
| `"this is an image of X"` | 60.00% | 33.74% |
| `"X presented in image"` | **63.87%** | **40.74%** |
| `"a photo of X"` | 57.48% | 40.28% |

## 2. Few-shot linear probe (ViT fully frozen, only classifier head trained, 20 epochs)

| Shots per class | BiomedCLIP | PubMedCLIP |
|---|---|---|
| 1 | 67.38% | 57.25% |
| 2 | 66.22% | 65.75% |
| 4 | 67.50% | 65.29% |
| 8 | 79.44% | 62.78% |
| 16 | 82.41% | 67.56% |

A second, independent run of the BiomedCLIP 16-shot setting (used as the
`unfreeze_last_n=0` baseline for the ablation below) produced **86.90%** —
the ~4.5-point gap between the two runs of the *same* configuration shows
that single-run numbers are not stable. The cause (random seed, or the
shared-model-state issue above) has not been isolated yet.

## 3. No-training baseline: class-mean prototype (BiomedCLIP only)

| Method | 16-shot accuracy |
|---|---|
| Linear probe (20 epochs training) | 86.90% |
| **Prototype (no training, class-mean + cosine similarity)** | **84.74%** |

Difference: 2.16 points — smaller than the 4.5-point difference between two
runs of the same configuration observed above. **The trained classifier is not clearly better than simply
averaging the few-shot embeddings**, which is the main evidence for this
project's central claim: pretrained feature quality, not few-shot training,
is the dominant factor.

## 4. Partial unfreezing of the visual backbone (BiomedCLIP only)

### 4a. Fixed unfreeze depth (last 2 blocks), varying shot count

| Shots per class | Frozen (baseline) | Unfreeze last 2 blocks |
|---|---|---|
| 1 | 67.38% | 52.59% (worse — overfitting) |
| 2 | 66.22% | 48.46% (worse — overfitting) |
| 4 | 67.50% | 68.32% (roughly even) |
| 8 | 79.44% | **86.66%** (clearly better) |

### 4b. Fixed shot count (16-shot), varying unfreeze depth

| `unfreeze_last_n` | 16-shot accuracy |
|---|---|
| 0 (frozen) | 86.90% |
| 1 | 85.61% |
| 2 | 85.26% |
| 4 | 83.34% |

**These two sub-experiments disagree** — (4a) suggests unfreezing helps once
shots ≥ 8, while (4b) shows a monotonic *decrease* with more unfrozen layers
at 16-shot. Given the ~4.5-point single-seed noise floor established in
section 3, this contradiction may be noise and/or an artifact of the shared-model-state
issue above, rather than a real effect —
**this whole section needs 3–5 seed repetition before drawing any
conclusion**, and is flagged as a known limitation rather than a finding.

## Take-away

| Comparison | Accuracy delta | Interpretation |
|---|---|---|
| Zero-shot → few-shot (any method) | +20–25 points | Having a handful of task-specific reference images/prompts helps a lot |
| No training → trained linear probe | ~2 points | Training the classifier barely matters once you have the reference images |
| Frozen → partially unfrozen backbone | Inconsistent across runs, ±5 points | Not yet conclusively different from noise |

This pattern is consistent across both models tested: **BiomedCLIP's much
larger pretraining corpus (PMC-15M, ~15M pairs) gives it a better-separated
feature space than PubMedCLIP's (ROCO, ~80K pairs)**, and that gap in
pretrained feature quality — not anything done at fine-tuning time — is what
explains most of the accuracy difference between the two models.
