"""Matplotlib figures for the demo. Dark-neutral palette so they sit well in the UI."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .config import DATASETS, BACKBONES, pretty

# Clinical palette: paper-white ground, green-biased neutrals, teal accent - matched to
# the app shell. The base/novel pair (#12a594 teal vs #e2711d orange) clears the CVD and
# normal-vision separation gates all-pairs; every bar also carries a direct value label,
# which is the relief the contrast warning requires.
INK = '#0f2e28'
MUTED = '#4d6b62'
GRID = '#d9e7e0'
PAPER = '#ffffff'
BASE_C = '#0b6e63'
NOVEL_C = '#e2711d'
ACCENT = '#12a594'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # validated categorical order

plt.rcParams.update({
    'figure.dpi': 110, 'font.size': 9, 'axes.grid': True, 'grid.alpha': .55,
    'grid.color': GRID, 'axes.axisbelow': True, 'axes.edgecolor': GRID,
    'axes.labelcolor': MUTED, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
    'savefig.facecolor': PAPER,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def session_curve(history, dataset):
    """history: list of dicts with keys n_seen, acc, balanced, base_acc, novel_acc."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if not history:
        ax.text(.5, .5, 'Start a session to see the curve', ha='center', va='center',
                color=MUTED, transform=ax.transAxes)
        ax.set_axis_off(); fig.tight_layout(); return fig

    x = list(range(len(history)))
    labels = [f"S{h['n_seen'] - DATASETS[dataset]['n_base']}" for h in history]
    labels[0] = 'base'
    ax.plot(x, [h['acc'] for h in history], '-o', color=ACCENT, lw=2.2, ms=6,
            label='overall accuracy')
    ax.plot(x, [h['balanced'] for h in history], '--s', color=NOVEL_C, lw=1.8, ms=5,
            label='balanced accuracy')
    bb = [h['base_acc'] for h in history]
    nn = [h['novel_acc'] for h in history]
    ax.plot(x, bb, ':', color=BASE_C, lw=1.8, label='base classes')
    if any(v is not None for v in nn):
        xs = [i for i, v in enumerate(nn) if v is not None]
        ax.plot(xs, [nn[i] for i in xs], ':', color=NOVEL_C, lw=1.8, label='novel classes')
    for i, h in enumerate(history):
        ax.annotate(f"{h['acc']:.1f}", (i, h['acc']), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('%'); ax.set_ylim(0, 105)
    ax.set_title('Accuracy after each incremental session', fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout(); return fig


def per_class_bars(recall, dataset, seen, base_cls, counts, k=5):
    meta = DATASETS[dataset]
    short = meta['short']
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.9 * len(short)), 3.3))
    xs, vals, cols, labs = [], [], [], []
    for i, c in enumerate(order):
        xs.append(i); vals.append(recall[c] if c in seen else 0)
        cols.append(BASE_C if c in base_cls else (NOVEL_C if c in seen else GRID))
        tag = '' if c in base_cls else ('\n5-shot' if c in seen else '\nnot yet seen')
        labs.append(f'{short[c]}{tag}')
    ax.bar(xs, vals, color=cols, width=.68)
    for x, v in zip(xs, vals):
        if v > 0:
            ax.text(x, v + 1.5, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel('recall %'); ax.set_ylim(0, 105)
    ax.set_title(f'Per-class recall  ·  teal = base session, orange = learned from {k} images',
                 fontsize=9.5)
    fig.tight_layout(); return fig


TEAL_RAMP = LinearSegmentedColormap.from_list(
    'clinical_teal', ['#ffffff', '#d8f0ea', '#8fd8cb', '#3fb7a5', '#12a594', '#0b6e63', '#063f39'])


def confusion_fig(cm, dataset, seen, base_cls, k=5):
    meta = DATASETS[dataset]
    short = meta['short']
    keep = sorted(seen)
    sub = cm[np.ix_(keep, keep)].astype(float)
    sub = sub / np.maximum(sub.sum(1, keepdims=True), 1) * 100
    n = len(keep)
    fig, ax = plt.subplots(figsize=(max(4.6, .78 * n + 1.8), max(4.0, .66 * n + 1.6)))
    im = ax.imshow(sub, cmap=TEAL_RAMP, vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if sub[i, j] >= 1:
                ax.text(j, i, f'{sub[i,j]:.0f}', ha='center', va='center', fontsize=7,
                        color='#0f2e28' if sub[i, j] < 45 else 'white')
    labs = [short[c] + ('' if c in base_cls else ' *') for c in keep]
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=55, ha='right', fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.grid(False)
    ax.set_title(f'Confusion (row-normalised %)  ·  * = learned from {k} images',
                 fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); return fig


def confidence_fig(conf, correct):
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    bins = np.linspace(min(conf), 1.0, 36)
    ax[0].hist(conf[correct], bins=bins, alpha=.7, color=ACCENT, density=True,
               label=f'correct (n={int(correct.sum())})')
    ax[0].hist(conf[~correct], bins=bins, alpha=.7, color='#d1495b', density=True,
               label=f'wrong (n={int((~correct).sum())})')
    ax[0].axvline(conf[correct].mean(), color=ACCENT, ls='--', lw=1.2)
    if (~correct).any():
        ax[0].axvline(conf[~correct].mean(), color='#d1495b', ls='--', lw=1.2)
    ax[0].set_xlabel('softmax confidence (tau = 16)'); ax[0].set_ylabel('density')
    ax[0].legend(fontsize=7.5); ax[0].set_title('Confidence: correct vs misclassified', fontsize=9.5)

    # risk-coverage: accuracy if we only answer on the most confident X%
    o = np.argsort(-conf); c = correct[o]
    covs = np.linspace(.05, 1, 40)
    accs = [c[:max(1, int(k * len(c)))].mean() * 100 for k in covs]
    ax[1].plot(covs * 100, accs, '-', color=ACCENT, lw=2.2)
    ax[1].axhline(correct.mean() * 100, color=MUTED, ls='--', lw=1,
                  label=f'answer everything: {correct.mean()*100:.1f}%')
    ax[1].set_xlabel('coverage — % of test set answered')
    ax[1].set_ylabel('accuracy on those answered %')
    ax[1].legend(fontsize=7.5)
    ax[1].set_title('Is the confidence usable for abstention?', fontsize=9.5)
    fig.tight_layout(); return fig


def backbone_compare(live, reported_advanced, fused, dataset):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    names, vals, cols, hatch = [], [], [], []
    for b, v in sorted(live.items(), key=lambda kv: -kv[1]):
        names.append(BACKBONES[b]['title'] + '\nfrozen (live)'); vals.append(v)
        cols.append(ACCENT); hatch.append('')
    if fused is not None:
        names.append('Fusion\n(live)'); vals.append(fused)
        cols.append(BASE_C); hatch.append('')
    for b, v in reported_advanced.items():
        if v is not None:
            names.append(f'{pretty(b)}\nPEFT (reported)'); vals.append(v)
            cols.append(NOVEL_C); hatch.append('//')
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=cols, width=.62)
    for bar, h in zip(bars, hatch):
        if h:
            bar.set_hatch(h); bar.set_edgecolor('white')
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.2, f'{v:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('final-session accuracy %'); ax.set_ylim(0, 105)
    ax.set_title(f"{DATASETS[dataset]['title']} — solid bars measured now; "
                 f"striped bars are stored results for a slower, retrained approach",
                 fontsize=9)
    fig.tight_layout(); return fig


def disagreement_fig(dis, reported=None):
    """Live pairwise disagreement, with the stored value from an earlier run marked
    alongside as a reference tick so the two can be read against each other."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    if not dis:
        ax.text(.5, .5, 'Select two or more backbones to compare',
                ha='center', va='center', color=MUTED, transform=ax.transAxes)
        ax.set_axis_off(); fig.tight_layout(); return fig
    keys = list(dis.keys()); vals = [dis[k] for k in keys]
    labs = [' vs\n'.join(BACKBONES[p]['title'].split()[0] for p in k.split('|')) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=ACCENT, width=.55, label='measured now')
    for xi, v in zip(x, vals):
        ax.text(xi, v + .8, f'{v:.1f}%', ha='center', fontsize=8.5, color=INK)

    shown = False
    for xi, k in zip(x, keys):
        rv = (reported or {}).get(k)
        if rv is not None:
            ax.plot([xi - .3, xi + .3], [rv, rv], color=NOVEL_C, lw=2,
                    label='stored result' if not shown else None)
            shown = True
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('% of test images predicted differently')
    ax.set_ylim(0, max(vals + [v for v in (reported or {}).values()]) * 1.3 + 4)
    if shown:
        ax.legend(fontsize=7.5)
    ax.set_title('Pairwise disagreement — high means complementary, so fusion can help',
                 fontsize=9.5)
    fig.tight_layout(); return fig
