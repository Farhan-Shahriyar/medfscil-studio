"""
The FSCIL engine: feature store, session state, fusion, and metrics.

The whole point of the demo is that an incremental session costs no gradient steps.
Adding a disease class is a mean over K feature vectors plus a TEEN shift, so every
interaction below is a NumPy operation measured in milliseconds.
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .config import FEATURES, DATASETS, BACKBONES, K_SHOT, SEED0, TAU


# ----------------------------------------------------------------- math
def power_transform(x, p=0.5):
    """Tukey's Ladder of Powers (Yang et al., ICLR 2021)."""
    return np.sign(x) * np.abs(x) ** p


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def simpleshot(train_f, test_f, center=True, tukey=True):
    """Centre on the training mean, Tukey, L2 - the pipeline the notebooks use."""
    tr, te = train_f.astype(np.float32), test_f.astype(np.float32)
    if center:
        m = tr.mean(axis=0, keepdims=True)
        tr, te = tr - m, te - m
    if tukey:
        tr, te = power_transform(tr), power_transform(te)
    return l2(tr), l2(te)


def teen_paper(p_raw, base_protos, alpha=0.5, tau=TAU):
    """Paper-correct TEEN (Wang et al., NeurIPS 2023): shift the noisy few-shot prototype
    toward a softmax-weighted blend of the base prototypes it actually resembles."""
    B = np.stack(base_protos)
    w = np.exp(tau * (B @ p_raw))
    w = w / w.sum()
    cal = p_raw + alpha * (w[:, None] * B).sum(axis=0)
    return cal / (np.linalg.norm(cal) + 1e-8)


def teen_naive(p_raw, seen_protos, beta=0.5):
    """The variant behind the published benchmark runs: shift toward the unweighted mean of
    every prototype seen so far - base classes AND novel classes added in earlier sessions.
    That growing reference set is what the benchmark notebooks use, and it is why this
    variant drifts: each new prototype is pulled toward the ones before it."""
    B = np.stack(seen_protos)
    cal = p_raw + beta * (B.mean(axis=0) - p_raw)
    return cal / (np.linalg.norm(cal) + 1e-8)


# ----------------------------------------------------------------- store
class FeatureStore:
    """Lazily memory-maps the cached features and caches the processed versions."""

    def __init__(self, root=FEATURES):
        self.root = root
        self._raw, self._proc = {}, {}

    def available_backbones(self, dataset):
        out = []
        for b in BACKBONES:
            p = os.path.join(self.root, b, f'{dataset}_test_feats.npy')
            if os.path.exists(p):
                out.append(b)
        return out

    def labels(self, dataset, split):
        key = ('lab', dataset, split)
        if key not in self._raw:
            any_b = self.available_backbones(dataset)
            if not any_b:
                raise FileNotFoundError(f'no features cached for {dataset}')
            self._raw[key] = np.load(
                os.path.join(self.root, any_b[0], f'{dataset}_{split}_labels.npy'))
        return self._raw[key]

    def raw(self, dataset, backbone, split):
        key = (dataset, backbone, split)
        if key not in self._raw:
            self._raw[key] = np.load(
                os.path.join(self.root, backbone, f'{dataset}_{split}_feats.npy'),
                mmap_mode='r')
        return self._raw[key]

    def processed(self, dataset, backbone, center=True, tukey=True):
        """Returns (train, test) after per-backbone SimpleShot + Tukey + L2."""
        key = (dataset, backbone, center, tukey)
        if key not in self._proc:
            tr = np.asarray(self.raw(dataset, backbone, 'train'), dtype=np.float32)
            te = np.asarray(self.raw(dataset, backbone, 'test'), dtype=np.float32)
            self._proc[key] = simpleshot(tr, te, center, tukey)
        return self._proc[key]

    def fused(self, dataset, backbones, center=True, tukey=True):
        """Feature-level (early) fusion: each backbone is normalised on its own scale
        first - otherwise the 2048-d ResNet vector simply drowns the 512-d CLIP one -
        then concatenated and re-normalised once."""
        if len(backbones) == 1:
            return self.processed(dataset, backbones[0], center, tukey)
        trs, tes = [], []
        for b in backbones:
            tr, te = self.processed(dataset, b, center, tukey)
            trs.append(tr); tes.append(te)
        return (l2(np.concatenate(trs, axis=1)), l2(np.concatenate(tes, axis=1)))


STORE = FeatureStore()


# ----------------------------------------------------------------- session
class Session:
    """One FSCIL run: base session, then novel classes added one at a time."""

    def __init__(self, dataset, backbones, fusion='concat', calib='paper',
                 center=True, tukey=True, seed=SEED0, store=STORE):
        self.dataset = dataset
        self.backbones = list(backbones)
        self.fusion = fusion if len(backbones) > 1 else 'concat'
        self.calib = calib
        self.center, self.tukey = center, tukey
        self.seed = seed
        self.store = store

        meta = DATASETS[dataset]
        self.n_cls = len(meta['names'])
        self.short = meta['short']
        self.train_l = store.labels(dataset, 'train')
        self.test_l = store.labels(dataset, 'test')

        counts = np.bincount(self.train_l, minlength=self.n_cls)
        order = np.argsort(-counts).tolist()
        self.counts = counts
        self.base_cls = order[:meta['n_base']]
        self.novel_order = order[meta['n_base']:]

        self.reset()

    # -------------------------------------------------- state
    def reset(self):
        # One generator per session, consumed sequentially as classes arrive - the benchmark
        # notebooks draw all their support sets from a single rng, so re-seeding per class
        # would give a different (and correlated) episode than the published protocol.
        self._rng = np.random.default_rng(self.seed)
        self.seen = list(self.base_cls)
        self.shots = {}           # class -> support indices actually used
        self.protos = {}          # per backbone: {backbone: {class: vector}}
        for b in self.backbones:
            tr, _ = self.store.processed(self.dataset, b, self.center, self.tukey)
            self.protos[b] = {c: self._proto(tr, self.train_l == c) for c in self.base_cls}
        if len(self.backbones) > 1 and self.fusion == 'concat':
            tr, _ = self.store.fused(self.dataset, self.backbones, self.center, self.tukey)
            self.protos['__fused__'] = {c: self._proto(tr, self.train_l == c)
                                        for c in self.base_cls}
        return self

    @staticmethod
    def _proto(feats, mask_or_idx):
        p = feats[mask_or_idx].mean(axis=0)
        return p / (np.linalg.norm(p) + 1e-8)

    def remaining(self):
        return [c for c in self.novel_order if c not in self.seen]

    # -------------------------------------------------- the incremental step
    def add_class(self, cls, k=K_SHOT, seed=None):
        """Learn one new disease class from k support images. No gradient steps."""
        if cls in self.seen:
            return self
        rng = self._rng if seed is None else np.random.default_rng(seed)
        idx = np.where(self.train_l == cls)[0]
        shot_idx = rng.choice(idx, size=min(k, len(idx)), replace=False)
        self.shots[cls] = shot_idx

        for space in list(self.protos.keys()):
            if space == '__fused__':
                tr, _ = self.store.fused(self.dataset, self.backbones, self.center, self.tukey)
            else:
                tr, _ = self.store.processed(self.dataset, space, self.center, self.tukey)
            p = self._proto(tr, shot_idx)
            if self.calib == 'paper':
                # paper-correct TEEN weights the BASE prototypes by similarity
                p = teen_paper(p, [self.protos[space][c] for c in self.base_cls])
            elif self.calib == 'naive':
                # the published variant averages over everything seen so far, which grows
                p = teen_naive(p, [self.protos[space][c] for c in self.seen])
            self.protos[space][cls] = p
        self.seen.append(cls)
        return self

    def add_all(self, k=K_SHOT, seed=None):
        for c in self.remaining():
            self.add_class(c, k, seed)
        return self

    # -------------------------------------------------- inference
    def _scores_single(self, space, test_f=None):
        protos = self.protos[space]
        ids = sorted(protos.keys())
        P = np.stack([protos[c] for c in ids])
        if test_f is None:
            if space == '__fused__':
                _, test_f = self.store.fused(self.dataset, self.backbones, self.center, self.tukey)
            else:
                _, test_f = self.store.processed(self.dataset, space, self.center, self.tukey)
        return ids, test_f @ P.T

    def scores(self):
        """Returns (class_ids, score matrix) under the configured fusion mode."""
        if len(self.backbones) == 1:
            return self._scores_single(self.backbones[0])
        if self.fusion == 'concat':
            return self._scores_single('__fused__')

        # late fusion: each backbone keeps its own prototypes and votes
        combined, ids = None, None
        for b in self.backbones:
            ids, S = self._scores_single(b)
            if self.fusion == 'score_avg':
                v = softmax(S * 5.0)
            elif self.fusion == 'borda':
                from scipy.stats import rankdata
                v = rankdata(S, axis=1)
            else:
                raise ValueError(self.fusion)
            combined = v if combined is None else combined + v
        return ids, combined

    def predict(self):
        ids, S = self.scores()
        return np.array(ids)[np.argmax(S, axis=1)], ids, S

    # -------------------------------------------------- metrics
    def evaluate(self, only_seen=False):
        preds, ids, S = self.predict()
        mask = np.isin(self.test_l, self.seen) if only_seen else np.ones(len(self.test_l), bool)
        y, p = self.test_l[mask], preds[mask]
        base_m = np.isin(y, self.base_cls)
        novel_seen = [c for c in self.seen if c not in self.base_cls]
        nov_m = np.isin(y, novel_seen)
        return dict(
            acc=float(accuracy_score(y, p) * 100),
            balanced=float(balanced_accuracy_score(y, p) * 100),
            base_acc=float(accuracy_score(y[base_m], p[base_m]) * 100) if base_m.any() else None,
            novel_acc=float(accuracy_score(y[nov_m], p[nov_m]) * 100) if nov_m.any() else None,
            n_eval=int(mask.sum()), n_seen=len(self.seen),
            preds=preds, ids=ids, scores=S, mask=mask,
        )

    def per_class_recall(self):
        preds, _, _ = self.predict()
        cm = confusion_matrix(self.test_l, preds, labels=list(range(self.n_cls)))
        return cm.diagonal() / np.maximum(cm.sum(1), 1) * 100, cm

    def confidence(self):
        """tau-scaled softmax confidence - the same temperature TEEN itself uses."""
        preds, ids, S = self.predict()
        P = softmax(S * TAU)
        conf = P.max(1)
        correct = preds == self.test_l
        return conf, correct, preds


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------- comparisons
def disagreement(dataset, backbones, calib='paper', store=STORE):
    """Pairwise prediction disagreement between single-backbone sessions - the
    orthogonality measure that justifies fusing them."""
    preds = {}
    for b in backbones:
        s = Session(dataset, [b], calib=calib, store=store).add_all()
        preds[b], _, _ = s.predict()
    out = {}
    for i in range(len(backbones)):
        for j in range(i + 1, len(backbones)):
            a, c = backbones[i], backbones[j]
            out[f'{a}|{c}'] = float(np.mean(preds[a] != preds[c]) * 100)
    return out, preds


def oracle_gain(dataset, backbones, store=STORE):
    """Share of the test set that at least one backbone gets right - the ceiling any
    fusion rule could reach, and the honest bound on what fusion is worth."""
    _, preds = disagreement(dataset, backbones, store=store)
    y = store.labels(dataset, 'test')
    any_right = np.zeros(len(y), bool)
    for b in backbones:
        any_right |= (preds[b] == y)
    singles = {b: float((preds[b] == y).mean() * 100) for b in backbones}
    return float(any_right.mean() * 100), singles
