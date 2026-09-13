# MedFSCIL Studio

An interactive demonstration of the thesis' backbone study: **few-shot class-incremental
learning for medical disease classification**, built on the same frozen features, the same
protocol, and the same numbers as the reports.

The demo exists to make one thing tangible. Adding a new disease class to this system is a
mean over five feature vectors plus a TEEN calibration shift — **no gradient steps, no
retraining**. In the UI that is a sub-millisecond operation you watch happen.

---

## What it shows

The app is written to stand alone — someone can open it cold and understand it without
reading anything else. Explanation lives behind collapsible panels so the default view
stays uncluttered.

| Tab | What you do | What it demonstrates |
|---|---|---|
| **Teach a disease** | The model knows only the common diseases. Press *Teach the next disease* and it acquires one from a handful of examples. Every control is live. | The core claim, with a timer: acquiring a class is one averaging step, no training. |
| **Which model to use** | Pick a collection; get a verdict, a table and a chart. | That the right pipeline depends on the domain gap — freeze when the model is already aligned, retrain when it is not, and neither when the input resolution has destroyed the signal. |
| **Combining models** | Combine models, then look at the images they read differently. | Why fusion works (disagreement) and how much is still unclaimed (the oracle ceiling). |
| **Where it goes wrong** | Examine a configuration's mistakes. | The deficiency analysis: dominant confusions, certainty on wrong answers, risk–coverage, and the overconfidence flag. |
| **Reference** | Look things up. | Collections, models, and the stored five-run averages. |

## Faithfulness

The engine reproduces the published benchmark **exactly** under the same configuration
(frozen backbone, SimpleShot + Tukey + L2, naive TEEN, K=5, seeds 42–46):

| Dataset | ResNet-50 here / published | BiomedCLIP here / published | DINOv2 here / published |
|---|---|---|---|
| DermaMNIST | 51.84 / 51.84 | 49.80 / 49.79 | 53.17 / 52.91 |
| PathMNIST | 75.88 / 75.87 | 87.74 / 87.74 | 78.23 / 77.04 |
| BloodMNIST | 71.16 / 71.13 | 81.53 / 81.54 | 76.05 / 75.49 |
| RetinaMNIST | 50.00 / 50.00 | 49.80 / 49.75 | 52.90 / 51.50 |

Pairwise disagreement on DermaMNIST also reproduces the fusion report (52.2% and 55.8%
against its 52.22% and 55.76%), and the all-three fusion on PathMNIST reaches 89.25%
against the report's best-fusion 87.82%.

Linear-probe ceilings are measured here rather than quoted (`scripts/compute_ceilings.py`);
the ResNet-50 row reproduces `analysis/resnet` exactly (79.40 / 91.59 / 96.08 / 61.25).

Two details matter for that agreement, and both are easy to get wrong:

1. **Naive TEEN averages over every prototype seen so far**, base *and* previously added
   novel classes — not just the base ones. Paper-correct TEEN instead softmax-weights the
   base prototypes by similarity. They are different algorithms, and the published runs use
   the first.
2. **One RNG per session, consumed sequentially.** Re-seeding per class produces a
   different, correlated episode and shifts results by 1–3 pp.

## What is computed live, and what is quoted

Everything in tabs 1, 3 and 4 is computed in-session from cached frozen features.

The **advanced PEFT** results (bottleneck adapters, SupCon, EWC/rehearsal, textual priors)
are *trained* pipelines from the BiomedCLIP and DINOv3 reports. They are quoted, never
recomputed, and are marked as such wherever they appear — hatched bars in the router chart,
italic notes elsewhere.

`DINOv2 ViT-B/14` stands in for DINOv3, which is a gated Hugging Face repository. This is
the same fallback the team's `multidisease_6_best_fusion.ipynb` uses when no `HF_TOKEN` is
present; the architecture family and the pipeline are identical.

---

## Setup

```bash
pip install -r requirements.txt
```

Then build the caches. Feature extraction streams the compressed `.npz` files directly, so
the 12.6 GB PathMNIST archive never has to fit in RAM:

```bash
python scripts/extract_features.py --backbone resnet50
python scripts/extract_features.py --backbone dinov2
python scripts/extract_features.py --backbone biomedclip
python scripts/build_thumbs.py --split test  --size 160
python scripts/build_thumbs.py --split train --size 128 --novel-only
```

On an RTX 3050 the three backbones take roughly 10 / 25 / 25 minutes for all four datasets.
`DATA_DIR` at the top of `scripts/extract_features.py` points at the MedMNIST-224 `.npz`
files; change it if yours live elsewhere.

## Run

```bash
python app.py
```

`--share` gives a temporary public link; `--port` changes the port. The app needs no GPU —
every interaction reads cached features and does NumPy arithmetic.

## Layout

```
demo/
├── app.py                      the Gradio application
├── medfscil/
│   ├── config.py               datasets, backbones, and the published results
│   ├── engine.py               FSCIL session, TEEN, fusion, metrics
│   ├── router.py               the domain-gap routing policy
│   ├── images.py               thumbnail access
│   └── plots.py                figures
├── scripts/
│   ├── extract_features.py     frozen features, all backbones, all datasets
│   └── build_thumbs.py         JPEG thumbnail packs for the UI
├── features/                   cached .npy features (large, not in git)
└── thumbs/                     cached JPEG packs (not in git)
```

## Deploying

The app is CPU-only at runtime, so a free Hugging Face Space works. Two things to trim
first, since `features/` and `thumbs/` are a few GB as built:

- Keep **test** features for all backbones but drop **train** features for the base classes,
  storing their prototypes instead — base prototypes are means over the full training set
  and never change.
- Keep train features only for the novel classes, which are the only ones that can be drawn
  as support. The thumbnail packs already follow this rule.

That brings the pack under roughly 150 MB with no change in behaviour.

## Interaction model

*Teach a disease* is fully reactive: collection, model, combination rule, correction
method, examples-per-disease and the draw seed all rebuild the session the moment they
change, so nothing on screen is ever stale and no control is silently inert. Sliders fire
on release rather than on drag. *Combining models* and *Where it goes wrong* use explicit
buttons because their computations take seconds on PathMNIST.

Nothing in the interface refers to an external document. Figures are either measured on
the spot or labelled as stored reference values; `GUIDE.md` is the operator's walkthrough,
not a prerequisite for using the app.

## Caveats

- Research prototype for a thesis demonstration. **Not a medical device, not a diagnostic
  tool.**
- The demo runs a single seed that you control. The published figures are 5-seed means, so
  small differences are expected — and for the tiny novel classes (DermaMNIST `df`,
  RetinaMNIST `g4`) the seed genuinely matters by ±10 pp. That variance is itself one of the
  study's findings, so the seed is exposed as a slider rather than hidden.
