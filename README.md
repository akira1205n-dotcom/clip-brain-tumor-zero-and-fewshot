# Zero-shot & Few-shot Brain Tumor MRI Classification with Medical CLIP Models

**English** | [日本語](README.ja.md)

This repository contains a series of experiments exploring how well pretrained
medical vision-language models (**BiomedCLIP**, **PubMedCLIP**) can classify
brain tumor MRI images with **zero training data (zero-shot)** and with
**only a handful of labeled images per class (few-shot)**.

The project started as a course exercise on CLIP-style vision-language models
and was extended into a set of controlled ablation experiments to answer one
central question:

> **When a CLIP-style model performs well on a downstream medical imaging
> task, is the improvement mainly driven by (a) the quality of the frozen
> pretrained features, or (b) the extra training done on top of them
> (linear probing, partial fine-tuning, etc.)?**

## TL;DR — Key Finding

Across every experiment, **the quality of the pretrained encoder dominates
downstream accuracy**. Additional training on top of a frozen backbone
(a linear classifier, or unfreezing a few ViT layers) only adds a small,
often statistically marginal, improvement — and can even hurt performance
when the number of labeled examples is very small.

| Setting | BiomedCLIP | PubMedCLIP |
|---|---|---|
| Zero-shot (best prompt) | 63.87% | 40.74% |
| Few-shot, 16-shot, linear probe | 82.41% | 67.56% |
| Few-shot, 16-shot, **no training** (class-mean prototype) | 84.74% | — |

The gap between "no training, just averaging features" (84.74%) and
"training a linear layer for 20 epochs" (86.90% in a separate run) is about
2 points, while two separate runs of the same linear-probe configuration
differed by more than that (82.41% vs 86.90%). The two methods are therefore
not distinguishable in these experiments, which is consistent with **the
pretrained backbone doing most of the work**. (See
[Limitations](#limitations): part of the run-to-run variation may come from a
shared-model-state issue rather than random seeds alone.)

## Models Compared

| Model | Visual backbone | Text backbone | Pretraining data |
|---|---|---|---|
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | ViT-B/16 | PubMedBERT | PMC-15M (~15M biomedical image-text pairs) |
| [PubMedCLIP](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32) | ViT-B/32 | CLIP text encoder (fine-tuned) | ROCO (~80K radiology image-caption pairs) |

Both are CLIP-style dual-encoder models: an image encoder and a text encoder
are trained with a contrastive objective so that matching image-text pairs
end up close together in a shared embedding space. See
[`docs/background.md`](docs/background.md) for a from-scratch explanation of
how ViT, BERT, and CLIP's contrastive pretraining work.

## Dataset

BTMRI (Brain Tumor MRI) — 4 classes, one folder per class:
`glioma_tumor`, `meningioma_tumor`, `normal_brain`, `pituitary_tumor`.

The dataset was distributed as `BTMRI.zip` together with the original course
exercise, along with a `split_BTMRI.json` file that defines the fixed
train / val / test split used by the few-shot experiments. A similar public
brain tumor MRI dataset exists on Kaggle, but its folder names and split
differ from this one, so please check the source and license of the data you
use before redistributing anything.

The raw MRI images are **not included in this repository** (dataset license /
size). Download it yourself and point `data_dir` in the notebooks to your
local copy. See the notebook's setup cell for the expected folder layout.

## Experiments

### 1. Zero-shot classification
No training at all. Each class name is embedded into a text prompt
(3 templates were compared), encoded with the text encoder, and compared via
cosine similarity against the image embedding. The prediction is simply the
closest matching class-prompt.

**Finding:** prompt wording matters — "`X presented in image`" style
templates outperformed the generic "`a photo of X`" template by ~6 points on
BiomedCLIP, likely because it better matches the language style of the
biomedical literature the model was pretrained on.

### 2. Few-shot linear probe
The image encoder is **fully frozen**; only a new `nn.Linear(512, num_classes)`
head is trained on 1 / 2 / 4 / 8 / 16 images per class.

**Finding:** BiomedCLIP's accuracy climbs steadily and substantially with more
shots (67% → 82%), while PubMedCLIP barely moves (57% → 67%, non-monotonic) —
consistent with its much smaller and less diverse pretraining corpus.

### 3. No-training baseline (class-mean prototype)
Instead of training a classifier, each class is represented by the mean of
its few-shot image embeddings, and test images are classified by nearest
prototype (cosine similarity) — i.e. the same "compare-and-pick-the-closest"
logic as zero-shot, but using averaged image features instead of a text
prompt as the reference vector.

**Finding:** this training-free baseline (84.74% at 16-shot) is nearly on par
with the trained linear probe, showing that most of the accuracy comes from
the frozen backbone's already well-separated feature space, not from the
20 epochs of linear-layer training.

### 4. Partial unfreezing of the visual backbone
The classifier head is kept, but the last *N* transformer blocks of the ViT
are also unfrozen and trained jointly.

**Finding (preliminary — single run, needs repetition):** unfreezing hurts
accuracy at very low shot counts (1–2 shot; likely overfitting, since a
couple of ViT blocks already contain millions of trainable parameters) and
only starts to help once there is enough data (~8 shot+). A follow-up sweep
over unfreeze depth (0/1/2/4 layers) at a fixed 16-shot did **not** show a
clean, monotonic trend — this part of the study needs multi-seed repetition
before drawing firm conclusions (see [Limitations](#limitations)).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_biomedclip_zero_few_shot.ipynb   # main experiment notebook
│   └── 02_pubmedclip_zero_few_shot.ipynb   # PubMedCLIP comparison
├── results/
│   └── results_summary.md                  # all accuracy numbers in one place
└── docs/
    └── background.md                       # ViT / BERT / CLIP contrastive pretraining primer
```

## Setup

```bash
pip install -r requirements.txt
```

Then open either notebook in Jupyter / Google Colab. Each notebook expects
the Brain Tumor MRI Dataset to be available locally — update the `data_dir`
/ `root_folder` variables in the setup cell to point at your copy.

## Limitations

- Most few-shot numbers come from a **single random seed**. Section 4
  (partial unfreezing) in particular showed contradictory trends between two
  separate runs at 16-shot, and the difference between "no training" and
  "trained linear probe" (2–4 points) is comparable to the observed
  run-to-run differences. **Multi-seed repetition (3–5 seeds per setting, mean ± std) is
  planned but not yet complete** — treat single-run numbers as indicative,
  not conclusive.
- **Shared-model-state issue (found after the experiments were run).**
  `LinearProbe(model, ...)` stores a reference to the global `model`, not a
  copy. In runs where ViT blocks were unfrozen, their weights were modified
  in place, and later runs in the same session started from these modified
  weights instead of the original pretrained ones. Runs that keep the backbone
  fully frozen (the original 1–16-shot linear-probe curves and the PubMedCLIP
  runs) are unaffected. The unfreezing results (section 4) — and possibly the
  `unfreeze_last_n=0` baseline and the prototype result, if they were run
  afterwards in the same session — need to be re-run with the pretrained
  weights restored before every run. Until then, the run-to-run gaps quoted
  above should not be attributed purely to random seeds.
- Batch size is set to the full few-shot training set
  (`train_batch_size = shot_num * num_classes`), so every configuration
  performs exactly one gradient update per epoch (20 updates in total for
  20 epochs) regardless of shot count. The "more shots -> higher accuracy"
  trend is therefore not confounded by the number of update steps. However,
  an explicit ablation with a decoupled, fixed batch size has not been run
  (see `docs/background.md`, section 6).

## Planned Next Steps

- **LoRA** (parameter-efficient fine-tuning) as a middle ground between
  "linear probe only" (too few trainable params) and "unfreeze whole
  blocks" (too many, overfits on tiny shot counts).
- **Knowledge distillation**: use BiomedCLIP's output distribution as a soft
  target to see whether it can partially compensate for PubMedCLIP's smaller
  pretraining corpus.
- Repeat all few-shot experiments over multiple random seeds and report
  mean ± std, per the [Limitations](#limitations) section.

## Acknowledgements / References

- Zhang et al., *BiomedCLIP: a multimodal biomedical foundation model
  pretrained from fifteen million scientific image-text pairs*, 2023.
- Eslami et al., *PubMedCLIP: How Much Does CLIP Benefit Visual Question
  Answering in the Medical Domain?*, 2021.
- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision (CLIP)*, ICML 2021.
- BTMRI brain tumor MRI dataset, as provided with the course exercise.

This started as a graduate-level exercise notebook and was extended with the
ablation studies described above.
