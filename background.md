# Background: How CLIP-style Models Work

This is a from-scratch explanation of the mechanics behind BiomedCLIP /
PubMedCLIP, written for anyone reading this repo without prior CLIP
background. It also documents a few implementation details of this specific
codebase that are easy to miss.

## 1. The image encoder (Vision Transformer)

A 224×224 image is split into non-overlapping 16×16 (or 32×32 for
PubMedCLIP) patches. Each patch is flattened and linearly projected into a
vector (in practice this "slice + flatten + project" step is implemented as
a single strided convolution, `Conv2d(3, hidden_dim, kernel_size=patch,
stride=patch)`).

A learnable `[CLS]` token is prepended to the sequence of patch vectors, and
a position embedding is added to each vector so the model knows where each
patch sat in the original image. The full sequence is passed through several
Transformer blocks (self-attention + MLP, repeated). After the final block,
the `[CLS]` token's vector is taken as the whole image's representation and
projected (via a small linear "head") down to the shared embedding
dimension (512 in both models used here).

## 2. The text encoder (BERT-style)

A sentence is tokenized (WordPiece/BPE), each token is looked up in an
embedding table, a `[CLS]` token is prepended, position embeddings are
added, and the sequence goes through the same kind of Transformer stack.
The `[CLS]` token's final vector, again projected to 512-d, represents the
sentence.

BiomedCLIP uses **PubMedBERT** — a BERT variant trained from scratch on
PubMed abstracts/full text, with a vocabulary built from biomedical
literature rather than general web text. This is why it handles terms like
"glioma" or "histopathology" more naturally than a general-purpose BERT.

## 3. Contrastive pretraining

Given a batch of *N* matched (image, caption) pairs:

1. Encode all *N* images → `I_1 ... I_N` (512-d each).
2. Encode all *N* captions → `T_1 ... T_N` (512-d each).
3. Compute the *N × N* similarity matrix `I_i · T_j` (cosine similarity ×
   a learned temperature, `logit_scale`).
4. Treat each row as a classification problem: for image `i`, the correct
   answer is caption `i` (the diagonal entry). Cross-entropy loss pushes
   diagonal similarities up and off-diagonal similarities down.
5. Backpropagate through **both** encoders and update their weights.

Repeated over millions of image-caption pairs, this teaches both encoders to
place semantically matching image/text pairs close together in the shared
512-d space — without ever being told "this is class X". This is what makes
zero-shot classification possible later: no fixed label set was ever baked
into the model.

## 4. Zero-shot classification

At inference time, for a new classification task with classes
`[c_1, ..., c_k]`:

1. Turn each class name into a sentence with a template, e.g.
   `"this is an image of {c}"`.
2. Encode all *k* prompts with the (frozen) text encoder → `T_1 ... T_k`.
3. Encode the query image with the (frozen) image encoder → `I`.
4. Predict `argmax_j (I · T_j)`.

No training happens here — the text encoder has never seen these exact
class names before, but because it learned general biomedical language
understanding during pretraining, it can still produce a "reasonable"
embedding for a new sentence. Prompt wording affects how close that
embedding lands to what the image encoder actually produces for matching
images, which is why prompt engineering measurably changes accuracy (see
`notebooks/01_biomedclip_zero_few_shot.ipynb`, section 10).

## 5. Few-shot linear probe

```python
class LinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False        # freeze everything
        self.classifier = nn.Linear(512, num_classes)   # new, trainable

    def forward(self, x):
        features = self.clip_model.encode_image(x)       # frozen forward pass
        features = F.normalize(features, dim=-1)
        return self.classifier(features)                 # trainable head
```

**"Frozen" does not mean "skipped"** — the image encoder still runs its full
forward pass on every image (patchify → transformer blocks → `[CLS]` →
projection), it just never receives a gradient update. Only the new linear
layer (512 × num_classes + bias — a few thousand parameters, versus the ~86M
in ViT-B) is trained with a handful of labeled images per class.

Because this linear layer has so few parameters, what it "learns" is not
anything about individual images — it's simply a rough direction, per class,
in the 512-d embedding space (one row of the weight matrix per class). If
the frozen encoder already places same-class images near each other and
different-class images far apart, this is an easy problem regardless of how
few examples you give it. This is the basis for the "no-training prototype
baseline" ablation below.

## 6. A caveat about the few-shot training loop in this repo

```python
train_batch_size = shot_num * num_classes
```

Batch size is set equal to the entire few-shot training subset, so **there is
exactly one gradient update per epoch**, regardless of shot count. With a
fixed `epochs=20` across all shot counts, this means every configuration
receives the same number of gradient steps (20) — so the "more shots → higher
accuracy" trend reflects genuinely more/varied training *data* per step, not
more training *steps*. (This was verified analytically rather than by an
explicit ablation — a good thing to double check if you extend this repo.)

## 7. Prototype (no-training) baseline

```python
prototype[c] = mean of L2-normalized image features for class c (few-shot images)
prediction   = argmax_c ( test_image_feature · prototype[c] )
```

This reuses exactly the same "compare embeddings via cosine similarity,
pick the closest" logic as zero-shot — the only difference is that the
reference vector for each class comes from averaging a handful of real
image embeddings instead of encoding a text prompt. No gradient descent, no
new parameters. See `results/results_summary.md` for how close this gets to
the trained linear probe.

## 8. Partial unfreezing

```python
blocks = clip_model.visual.trunk.blocks     # timm-style ViT inside open_clip
for block in list(blocks)[-unfreeze_last_n:]:
    for p in block.parameters():
        p.requires_grad = True
```

Unfreezing the last *N* transformer blocks allows the encoder itself to
adapt to the downstream task, not just the linear head on top. Deeper
(later) blocks tend to encode more task/semantic-specific features, while
early blocks encode generic low-level visual patterns — this is why only
the *last* blocks are unfrozen rather than the first ones. The trade-off:
each unfrozen block adds millions of trainable parameters (a single ViT-B
block has roughly 7M), so with very few shots this can overfit badly (see
`results/results_summary.md`).
