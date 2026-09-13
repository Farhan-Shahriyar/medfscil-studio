"""
Static configuration and the published results from the three backbone reports.

Everything the demo computes live is derived from the cached features; everything in
REPORTED / ADVANCED / FUSION / DISAGREEMENT below is quoted from the team's own
benchmark documents and is displayed as *evidence*, clearly labelled as such, never
mixed into a live number.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, 'features')
THUMBS = os.path.join(ROOT, 'thumbs')
# Where the MedMNIST-224 .npz archives live. Override with the MEDMNIST_DIR environment
# variable; otherwise a `data/` folder beside this project is assumed.
DATA_DIR = os.environ.get('MEDMNIST_DIR') or os.path.join(ROOT, 'data')

K_SHOT = 5
SEED0 = 42
TAU = 16.0

# ----------------------------------------------------------------- datasets
DATASETS = {
    'dermamnist': dict(
        title='DermaMNIST', modality='Dermatoscopy (macroscopic skin photography)',
        n_base=4, source='HAM10000',
        names=['actinic keratoses / intraepithelial carcinoma', 'basal cell carcinoma',
               'benign keratosis-like lesions', 'dermatofibroma', 'melanoma',
               'melanocytic nevi', 'vascular lesions'],
        short=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
        domain='out-of-domain',
        domain_note='None of these models saw photographs of skin lesions while being '
                    'built, so all of them are working outside what they know.'),
    'pathmnist': dict(
        title='PathMNIST', modality='Colorectal histopathology (H&E stained slides)',
        n_base=5, source='NCT-CRC-HE-100K',
        names=['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle',
               'normal colon mucosa', 'cancer-associated stroma',
               'colorectal adenocarcinoma epithelium'],
        short=['adipose', 'background', 'debris', 'lymph', 'mucus', 'muscle',
               'normal-mucosa', 'stroma', 'tumor-epi'],
        domain='in-domain',
        domain_note='Stained tissue slides appear constantly in the medical literature one '
                    'of these models was built from, so it arrives already fluent in them.'),
    'bloodmnist': dict(
        title='BloodMNIST', modality='Peripheral blood cell microscopy',
        n_base=4, source='Barcelona blood cell dataset',
        names=['basophil', 'eosinophil', 'erythroblast', 'immature granulocytes',
               'lymphocyte', 'monocyte', 'neutrophil', 'platelet'],
        short=['baso', 'eos', 'erythro', 'ig', 'lymph', 'mono', 'neutro', 'platelet'],
        domain='in-domain',
        domain_note='Blood-cell microscopy is well covered by one of the models. Note that '
                    '"immature granulocytes" is really three different cell types filed '
                    'under one label, which no single average can represent.'),
    'retinamnist': dict(
        title='RetinaMNIST', modality='Retinal fundus photography (DR severity grading)',
        n_base=3, source='DeepDRiD',
        names=['grade 0 (no DR)', 'grade 1', 'grade 2', 'grade 3', 'grade 4 (severe)'],
        short=['g0', 'g1', 'g2', 'g3', 'g4'],
        domain='resolution-bottlenecked',
        domain_note='Grading severity depends on tiny spots a few pixels wide. Shrinking the '
                    'photograph to a fixed small size erases them before any model sees it. '
                    'The grades are also a scale rather than separate categories.'),
}

# ----------------------------------------------------------------- backbones
BACKBONES = {
    'resnet50': dict(
        title='ResNet-50', family='CNN', dim=2048,
        pretrain='ImageNet-1k (IMAGENET1K_V2)',
        note='Supervised natural-image CNN. Post-ReLU global-average-pooled features: '
             'sparse, right-skewed, and in need of the full SimpleShot + Tukey recipe.'),
    'dinov2': dict(
        title='DINOv2 ViT-B/14', family='Self-supervised ViT', dim=1536,
        pretrain='LVD-142M (self-supervised, natural images)',
        note='CLS + patch-GAP concatenated. LayerNorm output is dense and symmetric, so the '
             'CNN-oriented feature transforms are redundant here. Stands in for DINOv3, '
             'which is a gated repository (same architecture family and pipeline).'),
    'biomedclip': dict(
        title='BiomedCLIP ViT-B/16', family='Vision-language (CLIP)', dim=512,
        pretrain='PMC-15M (15M biomedical figure-caption pairs)',
        note='The only backbone with biomedical pre-training. Its advantage is real but '
             'strictly modality-specific - it tracks what PMC-15M actually contains.'),
}

# ------------------------------------------------- published results (evidence only)
# Frozen backbone + NCM + TEEN, 5-shot, 5 seeds. Source: fscil_benchmark_results.pdf
REPORTED_FROZEN = {
    'resnet50':   {'dermamnist': 51.84, 'pathmnist': 75.87, 'bloodmnist': 71.13, 'retinamnist': 50.00},
    'biomedclip': {'dermamnist': 49.79, 'pathmnist': 87.74, 'bloodmnist': 81.54, 'retinamnist': 49.75},
    'dinov2':     {'dermamnist': 52.91, 'pathmnist': 77.04, 'bloodmnist': 75.49, 'retinamnist': 51.50},
    'dinov3':     {'dermamnist': 51.32, 'pathmnist': 80.96, 'bloodmnist': 82.74, 'retinamnist': 52.40},
}

# Advanced PEFT pipelines (adapters + SupCon + EWC/rehearsal + textual priors).
# These are TRAINED models: quoted from the reports, not recomputed by this demo.
REPORTED_ADVANCED = {
    'biomedclip': {'dermamnist': 68.04, 'pathmnist': 64.99, 'bloodmnist': 77.87, 'retinamnist': 54.80},
    'dinov3':     {'dermamnist': 74.37, 'pathmnist': 66.05, 'bloodmnist': 73.32, 'retinamnist': 55.45},
}

# Best fusion combination per dataset. Source: FSCIL_Backbone_Analysis_Report.pdf
REPORTED_FUSION = {
    'dermamnist':  (55.76, 'ResNet-50 + DINOv3 + BiomedCLIP'),
    'pathmnist':   (87.82, 'ResNet-50 + DINOv3 + BiomedCLIP'),
    'bloodmnist':  (86.89, 'DINOv3 + BiomedCLIP'),
    'retinamnist': (55.10, 'ResNet-50 + DINOv3'),
}

# Pairwise seed-0 prediction disagreement, Approach A. Higher = more complementary.
REPORTED_DISAGREEMENT = {
    'dermamnist':  {'resnet50|dinov2': 41.40, 'resnet50|biomedclip': 52.22, 'dinov2|biomedclip': 49.63},
    'pathmnist':   {'resnet50|dinov2': 24.78, 'resnet50|biomedclip': 26.81, 'dinov2|biomedclip': 20.19},
    'bloodmnist':  {'resnet50|dinov2': 26.98, 'resnet50|biomedclip': 27.33, 'dinov2|biomedclip': 23.27},
    'retinamnist': {'resnet50|dinov2': 41.25, 'resnet50|biomedclip': 42.00, 'dinov2|biomedclip': 42.50},
}

# Linear-probe ceilings on the SAME frozen features - the error-budget experiment.
# Fallbacks below are quoted (ResNet-50 from analysis/resnet; the DINOv3 row from the DINOv3
# report, which is NOT the same checkpoint as the DINOv2 that stands in for it here).
# scripts/compute_ceilings.py measures them per backbone and those values win when present.
_QUOTED_CEILING = {
    'resnet50': {'dermamnist': 79.40, 'pathmnist': 91.59, 'bloodmnist': 96.08, 'retinamnist': 61.25},
    'dinov3':   {'dermamnist': 82.50, 'pathmnist': 93.60, 'bloodmnist': 97.30, 'retinamnist': 64.20},
}


def _load_ceilings():
    path = os.path.join(ROOT, 'ceilings.json')
    measured = {}
    if os.path.exists(path):
        import json
        try:
            raw = json.load(open(path))
            measured = {b: {d: v['acc'] for d, v in dd.items()} for b, dd in raw.items()}
        except Exception:
            measured = {}
    out = {b: dict(v) for b, v in _QUOTED_CEILING.items()}
    for b, dd in measured.items():
        out.setdefault(b, {}).update(dd)
    return out


PROBE_CEILING = _load_ceilings()

# Confidence assigned to WRONG predictions by the trained advanced pipelines.
# Source: BioMedCLIP_Thesis_PreDefense_Report.pdf, which states these two explicitly.
# It gives no per-dataset figure for PathMNIST or RetinaMNIST, so neither is listed here -
# the report's claim for those is only the general ">94% confident while misclassifying".
ADVANCED_ERROR_CONFIDENCE = {
    'dermamnist': dict(advanced=92.36),
    'bloodmnist': dict(baseline=94.20, advanced=96.50),
}

# ----------------------------------------------------------------- router policy
ROUTING = {
    'in-domain': dict(
        verdict='Leave the model alone — just match against averages.',
        why='The model already tells these diseases apart on its own. Retraining it on a '
            'handful of examples bends it to fit those few and makes it worse at everything '
            'else.',
        evidence='On tissue slides, retraining part of the model dropped accuracy from 87.7% '
                 'to 65.0%. On blood cells one cell type went from being recognised most of '
                 'the time to almost never.',
        peft='harmful'),
    'out-of-domain': dict(
        verdict='Retrain part of the model — it is worth the cost here.',
        why='No model here has seen this kind of image before, so there is no existing skill '
            'to protect. Retraining builds something new rather than breaking something that '
            'worked.',
        evidence='On skin lesions, retraining part of the model lifted accuracy from 49.8% '
                 'to 68.0%, and from 51.3% to 74.4% for a second model.',
        peft='necessary'),
    'resolution-bottlenecked': dict(
        verdict='Neither helps — the images need to arrive at higher resolution.',
        why='The detail that decides the answer is thrown away when the image is shrunk, '
            'before any model looks at it. Nothing done afterwards can recover it.',
        evidence='Every model stalls near 50%. Even given all the training data, none can '
                 'pass about 61-69% here, against 96-98% on blood cells. Roughly two thirds '
                 'of the most severe cases are graded one step too low, whatever you change.',
        peft='insufficient'),
}

DISPLAY_NAME = {b: BACKBONES[b]['title'] for b in BACKBONES}
DISPLAY_NAME['dinov3'] = 'DINOv3 ViT-B/16'


def pretty(key):
    return DISPLAY_NAME.get(key, key)


DISCLAIMER = ('Research prototype — not a medical device and not a diagnostic tool. '
              'Nothing here should be used to make a clinical decision.')
