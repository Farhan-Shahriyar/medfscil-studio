"""
The domain-gap router.

The thesis' central claim across all three backbone branches is that there is no single
best pipeline: the right one is determined by how far the target modality sits from the
backbone's pre-training distribution. This module turns that claim into an actual
decision function, and reports the evidence behind each decision.

The routing *policy* is fixed by the study's findings (config.ROUTING). What is computed
live here is the evidence: which backbone wins on this dataset right now, how far apart
the backbones are, and how much headroom a better head could still recover.
"""
import numpy as np

from .config import (DATASETS, BACKBONES, ROUTING, REPORTED_ADVANCED,
                     REPORTED_FUSION, PROBE_CEILING)
from .engine import Session, disagreement, oracle_gain, STORE


def route(dataset, store=STORE):
    meta = DATASETS[dataset]
    policy = ROUTING[meta['domain']]
    available = store.available_backbones(dataset)

    live = {}
    for b in available:
        s = Session(dataset, [b], store=store).add_all()
        live[b] = s.evaluate()['acc']

    best_single = max(live, key=live.get) if live else None
    fused_acc = None
    if len(available) > 1:
        fs = Session(dataset, available, fusion='concat', store=store).add_all()
        fused_acc = fs.evaluate()['acc']

    dis, _ = (disagreement(dataset, available, store=store) if len(available) > 1 else ({}, {}))
    oracle, singles = (oracle_gain(dataset, available, store=store)
                       if len(available) > 1 else (None, live))

    ceiling = {b: PROBE_CEILING.get(b, {}).get(dataset) for b in available}
    ceiling = {k: v for k, v in ceiling.items() if v is not None}

    return dict(
        dataset=dataset, domain=meta['domain'], modality=meta['modality'],
        domain_note=meta['domain_note'], policy=policy,
        live=live, best_single=best_single, fused=fused_acc,
        disagreement=dis, oracle=oracle, ceiling=ceiling,
        reported_advanced={b: REPORTED_ADVANCED.get(b, {}).get(dataset) for b in REPORTED_ADVANCED},
        reported_fusion=REPORTED_FUSION.get(dataset),
    )


BADGE = {'in-domain': ('Already aligned', 'ok'),
         'out-of-domain': ('Far from pre-training', 'warn'),
         'resolution-bottlenecked': ('Signal lost at the input', 'bad')}


def decision_card(r):
    """Compact verdict block. Detail lives behind the 'Why' panel, not on screen."""
    meta = DATASETS[r['dataset']]
    label, tone = BADGE[r['domain']]
    return (f"<div class='decision {tone}'>"
            f"<div class='dk'>{meta['title']} · {r['modality']}</div>"
            f"<div class='db'>{label}</div>"
            f"<div class='dv'>{r['policy']['verdict']}</div></div>")


def evidence_rows(r):
    """One row per backbone: what it scores now, what its features could support, and the
    difference - which is the amount the class-mean head is throwing away."""
    rows = []
    for b, a in sorted(r['live'].items(), key=lambda kv: -kv[1]):
        ceil = r['ceiling'].get(b)
        rows.append([BACKBONES[b]['title'] + (' ★' if b == r['best_single'] else ''),
                     f'{a:.2f}%',
                     f'{ceil:.1f}%' if ceil else '—',
                     f'{ceil - a:.1f} pp' if ceil else '—'])
    if r['fused'] is not None:
        rows.append(['All combined', f"{r['fused']:.2f}%", '—', '—'])
    if r['oracle'] is not None:
        rows.append(['At least one is right', f"{r['oracle']:.2f}%", '—', '—'])
    return rows


def why_markdown(r):
    pol = r['policy']
    return (f"{pol['why']}\n\n**Evidence.** {pol['evidence']}\n\n"
            f"**About this modality.** {r['domain_note']}\n\n"
            f"**Reading the table.** *Score now* is this pipeline measured on the spot. "
            f"*Features could support* is what a plain linear classifier reaches on exactly "
            f"the same frozen features, given every training image — so the last column is "
            f"accuracy the class-mean head throws away, not accuracy the backbone never had. "
            f"★ marks the strongest single backbone here.")
