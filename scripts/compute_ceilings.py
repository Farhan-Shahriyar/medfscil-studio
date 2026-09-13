"""
Measure the linear-probe ceiling for every (backbone, dataset) pair.

This is the error-budget experiment from the ResNet report, generalised: a logistic
regression trained on the SAME frozen features, with all classes and their full training
data, bounds what those features can support at all. Comparing it against the FSCIL number
separates "the backbone cannot see it" from "the prototype head cannot exploit it".

Not FSCIL-legal - it is a diagnostic ceiling, and the UI labels it as one.

Writes demo/ceilings.json, which config.py picks up in preference to the quoted values.
"""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medfscil.config import DATASETS, BACKBONES
from medfscil.engine import STORE

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ceilings.json')
MAX_TRAIN = 25000     # subsample for tractability; PathMNIST has 90k


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for b in BACKBONES:
        res.setdefault(b, {})
        for d in DATASETS:
            if d in res[b]:
                print(f'{b}/{d}: cached {res[b][d]["acc"]:.2f}'); continue
            if b not in STORE.available_backbones(d):
                print(f'{b}/{d}: no features yet, skipping'); continue
            t0 = time.time()
            tr, te = STORE.processed(d, b)          # same SimpleShot + Tukey + L2 as FSCIL
            trl, tel = STORE.labels(d, 'train'), STORE.labels(d, 'test')
            rng = np.random.default_rng(0)
            idx = rng.choice(len(tr), min(MAX_TRAIN, len(tr)), replace=False)
            clf = LogisticRegression(max_iter=2000, C=10.0)
            clf.fit(tr[idx], trl[idx])
            p = clf.predict(te)
            res[b][d] = dict(acc=float(accuracy_score(tel, p) * 100),
                             balanced=float(balanced_accuracy_score(tel, p) * 100))
            print(f'{b}/{d}: probe {res[b][d]["acc"]:.2f} '
                  f'(balanced {res[b][d]["balanced"]:.2f})  [{time.time()-t0:.0f}s]', flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
