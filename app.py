"""
MedFSCIL Studio - teach a frozen vision backbone a new disease from a handful of images.

Run:  python app.py            (add --share for a temporary public link)

Design rules for this file:
  * The app stands alone. Nothing on screen refers to an external document; every figure
    is either measured live or labelled as a stored reference value.
  * Numbers and controls are the interface. Explanation lives behind "Why / How to read
    this" panels so the default view stays uncluttered.
  * Tab 1 is fully reactive - every control rebuilds the session immediately, so no
    control is ever silently inert. Tabs 3 and 4 use buttons, because those computations
    take seconds on the largest dataset.
"""
import argparse, time
from itertools import combinations, chain

import numpy as np
import gradio as gr

from medfscil.config import (DATASETS, BACKBONES, DISCLAIMER, K_SHOT, SEED0, pretty,
                             REPORTED_FROZEN, REPORTED_ADVANCED, REPORTED_FUSION,
                             REPORTED_DISAGREEMENT, ADVANCED_ERROR_CONFIDENCE)
from medfscil.engine import Session, STORE, disagreement, oracle_gain
from medfscil.images import THUMBS_PACK
from medfscil import plots
from medfscil.router import route, decision_card, evidence_rows, why_markdown

READY = [d for d in DATASETS if STORE.available_backbones(d)]
if not READY:
    raise SystemExit('No cached features found. Run scripts/extract_features.py first.')

FUSION_MODES = {'Join their features': 'concat',
                'Average their votes': 'score_avg',
                'Rank vote': 'borda'}
CALIB = {'Similarity-weighted (best)': 'paper',
         'Simple average': 'naive',
         'Off': 'none'}


# ================================================================= helpers
def backbone_choices(dataset):
    return [BACKBONES[b]['title'] for b in STORE.available_backbones(dataset)]


def to_keys(titles, dataset):
    t2k = {BACKBONES[b]['title']: b for b in STORE.available_backbones(dataset)}
    return [t2k[t] for t in (titles or []) if t in t2k]


def fmt_time(sec):
    ms = sec * 1000
    return f'{ms * 1000:.0f} µs' if ms < 1 else (f'{ms:.1f} ms' if ms < 100 else f'{ms:.0f} ms')


def cards(*items):
    """items: (label, value, sub) or (label, value, sub, tone)."""
    out = []
    for it in items:
        label, value, sub = it[0], it[1], it[2]
        tone = it[3] if len(it) > 3 else ''
        out.append(f"<div class='metric {tone}'><div class='k'>{label}</div>"
                   f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>")
    return "<div class='card-row'>" + ''.join(out) + "</div>"


def session_cards(ev, elapsed=None):
    items = [('Accuracy', f"{ev['acc']:.1f}%", f"{ev['n_eval']:,} test images"),
             ('Balanced', f"{ev['balanced']:.1f}%", 'all classes weighted equally'),
             ('Known from full data', f"{ev['base_acc']:.1f}%" if ev['base_acc'] is not None else '—',
              'the starting classes'),
             ('Known from a few images', f"{ev['novel_acc']:.1f}%" if ev['novel_acc'] is not None else '—',
              'the added classes'),
             ('Diseases known', f"{ev['n_seen']}", 'right now')]
    if elapsed is not None:
        items.append(('Time to learn', fmt_time(elapsed), 'no training', 'good'))
    return cards(*items)


def chips(sess, k):
    short = DATASETS[sess.dataset]['short']
    learned = [c for c in sess.seen if c not in sess.base_cls]
    html = ["<div class='chips'>"]
    for c in sess.base_cls:
        html.append(f"<span class='chip base'>{short[c]}</span>")
    for c in learned:
        html.append(f"<span class='chip novel'>{short[c]} · {k} images</span>")
    for c in sess.remaining():
        html.append(f"<span class='chip todo'>{short[c]}</span>")
    html.append("</div>")
    html.append("<div class='chip-key'><span class='chip base'></span>learned from full data"
                "<span class='chip novel'></span>learned from a few images"
                "<span class='chip todo'></span>not shown to the model yet</div>")
    return ''.join(html)


def _outputs(sess, hist, k, elapsed=None, gallery=None, banner=''):
    d = sess.dataset
    ev = sess.evaluate()
    rec, cm = sess.per_class_recall()
    gal = gr.update(value=gallery, visible=True) if gallery else gr.update()
    return (sess, hist, session_cards(ev, elapsed), banner, chips(sess, k),
            plots.session_curve(hist, d),
            plots.per_class_bars(rec, d, sess.seen, sess.base_cls, sess.counts, k),
            plots.confusion_fig(cm, d, sess.seen, sess.base_cls, k), gal)


# ================================================================= tab 1
def snapshot(sess):
    ev = sess.evaluate()
    return dict(n_seen=ev['n_seen'], acc=ev['acc'], balanced=ev['balanced'],
                base_acc=ev['base_acc'], novel_acc=ev['novel_acc'])


def build_session(dataset, bb_titles, fusion_label, calib_label, shots, seed):
    bbs = to_keys(bb_titles, dataset)
    if not bbs:
        return (None, [], "<div class='notice'>Pick at least one model to start.</div>",
                '', '', None, None, None, gr.update(value=None, visible=False))
    sess = Session(dataset, bbs, fusion=FUSION_MODES[fusion_label],
                   calib=CALIB[calib_label], seed=int(seed))
    n_new = len(sess.remaining())
    banner = (f"<div class='banner'>Ready. The model knows {len(sess.base_cls)} diseases and has "
              f"{n_new} more to learn — press <b>Teach the next disease</b>.</div>")
    out = _outputs(sess, [snapshot(sess)], int(shots), banner=banner)
    return out[:8] + (gr.update(value=None, visible=False),)


def learn_next(sess, hist, shots, dataset, bb_titles, fusion_label, calib_label, seed):
    if sess is None:                       # lazy start, so click order never matters
        built = build_session(dataset, bb_titles, fusion_label, calib_label, shots, seed)
        if built[0] is None:
            return built
        sess, hist = built[0], built[1]

    left = sess.remaining()
    if not left:
        banner = ("<div class='banner'>Every disease has been learned. Move the "
                  "<b>which images</b> slider and press reset to try a different draw.</div>")
        return _outputs(sess, hist, int(shots), banner=banner)

    c = left[0]
    t0 = time.perf_counter()
    sess.add_class(c, k=int(shots))
    elapsed = time.perf_counter() - t0

    imgs = THUMBS_PACK.many(sess.dataset, sess.shots[c], split='train')
    gal = [(im, f'{i + 1} of {len(imgs)}') for i, im in enumerate(imgs)]
    name = DATASETS[sess.dataset]['names'][c]
    banner = (f"<div class='banner good'>Learned <b>{name}</b> from {len(sess.shots[c])} images "
              f"in {fmt_time(elapsed)} — no training, no weight updates.</div>")
    return _outputs(sess, hist + [snapshot(sess)], int(shots), elapsed, gal or None, banner)


def learn_all(sess, hist, shots, dataset, bb_titles, fusion_label, calib_label, seed):
    out = learn_next(sess, hist, shots, dataset, bb_titles, fusion_label, calib_label, seed)
    if out[0] is None:
        return out
    while out[0].remaining():
        out = learn_next(out[0], out[1], shots, dataset, bb_titles,
                         fusion_label, calib_label, seed)
    return out


# ================================================================= tab 3
def fusion_report(dataset, bb_titles, fusion_label):
    bbs = to_keys(bb_titles, dataset)
    if not bbs:
        return "<div class='notice'>Pick at least one model.</div>", None, []
    ev = Session(dataset, bbs, fusion=FUSION_MODES[fusion_label]).add_all().evaluate()
    singles = {b: Session(dataset, [b]).add_all().evaluate()['acc']
               for b in STORE.available_backbones(dataset)}
    best_b = max(singles, key=singles.get)
    dis, _ = (disagreement(dataset, bbs) if len(bbs) > 1 else ({}, {}))
    orc, _ = (oracle_gain(dataset, bbs) if len(bbs) > 1 else (None, None))

    delta = ev['acc'] - singles[best_b]
    tone = 'good' if delta > 0.5 else ('' if abs(delta) <= 0.5 else 'bad')
    items = [('Combined', f"{ev['acc']:.1f}%", 'the models working together'),
             ('Best on its own', f"{singles[best_b]:.1f}%", BACKBONES[best_b]['title']),
             ('Difference', f'{delta:+.1f} pp', 'combined minus best single', tone)]
    if orc is not None:
        items.append(('Ceiling', f'{orc:.1f}%', 'at least one model is right', 'good'))
        items.append(('Still unclaimed', f"{orc - ev['acc']:.1f} pp", 'a better rule could win'))
    rows = [[BACKBONES[b]['title'], f'{v:.2f}%'] for b, v in
            sorted(singles.items(), key=lambda kv: -kv[1])]
    return (cards(*items), plots.disagreement_fig(dis, REPORTED_DISAGREEMENT.get(dataset, {})),
            rows)


def fusion_sweep(dataset):
    avail = STORE.available_backbones(dataset)
    combos = list(chain.from_iterable(combinations(avail, r) for r in range(1, len(avail) + 1)))
    label = {v: k for k, v in FUSION_MODES.items()}
    rows = []
    for combo in combos:
        for m in (['concat'] if len(combo) == 1 else list(FUSION_MODES.values())):
            ev = Session(dataset, list(combo), fusion=m).add_all().evaluate()
            rows.append([' + '.join(BACKBONES[b]['title'] for b in combo),
                         label[m] if len(combo) > 1 else 'on its own',
                         round(ev['acc'], 2), round(ev['balanced'], 2),
                         round(ev['base_acc'], 1), round(ev['novel_acc'], 1)])
    rows.sort(key=lambda r: -r[2])
    return rows


def disagreement_gallery(dataset, n_show):
    avail = STORE.available_backbones(dataset)
    if len(avail) < 2:
        return "<div class='notice'>Needs at least two models.</div>", gr.update(visible=False), []
    preds = {b: Session(dataset, [b]).add_all().predict()[0] for b in avail}
    y = STORE.labels(dataset, 'test')

    n_right = sum((preds[b] == y).astype(int) for b in avail)
    rescuable = np.where((n_right > 0) & (n_right < len(avail)))[0]
    if len(rescuable) == 0:
        return "<div class='notice'>The models agree on every image here.</div>", \
               gr.update(visible=False), []

    pick = np.random.default_rng(0).permutation(rescuable)[:int(n_show)]
    short = DATASETS[dataset]['short']
    code = {b: BACKBONES[b]['title'][0] for b in avail}
    gal, rows = [], []
    for n, i in enumerate(pick, 1):
        im = THUMBS_PACK.get(dataset, int(i), 'test')
        if im is None:
            continue
        marks = ' '.join(f"{code[b]}{'✓' if preds[b][i] == y[i] else '✗'}" for b in avail)
        gal.append((im, f'{n}. {short[y[i]]} · {marks}'))
        rows.append([n, short[y[i]]] + [short[preds[b][i]] +
                    (' ✓' if preds[b][i] == y[i] else '') for b in avail])
    head = cards(('Images one model gets right and another gets wrong',
                  f'{len(rescuable) / len(y) * 100:.1f}%',
                  f'{len(rescuable):,} of {len(y):,} test images', 'good'))
    return head, gr.update(value=gal, visible=True), rows


# ================================================================= tab 4
def error_report(dataset, bb_titles, fusion_label, n_show):
    bbs = to_keys(bb_titles, dataset)
    if not bbs:
        return "<div class='notice'>Pick at least one model.</div>", None, \
               gr.update(visible=False), [], None
    sess = Session(dataset, bbs, fusion=FUSION_MODES[fusion_label]).add_all()
    conf, correct, preds = sess.confidence()
    _, cm = sess.per_class_recall()
    y, short = sess.test_l, DATASETS[dataset]['short']

    wrong = np.where(~correct)[0]
    order = wrong[np.argsort(-conf[wrong])][:int(n_show)]
    gal = []
    for i in order:
        im = THUMBS_PACK.get(dataset, int(i), 'test')
        if im is not None:
            gal.append((im, f'{short[y[i]]} → {short[preds[i]]} · {conf[i]:.2f}'))

    hi = conf > 0.9
    err_hi = float((~correct[hi]).mean() * 100) if hi.any() else float('nan')
    head = cards(
        ('Sure and right', f'{conf[correct].mean():.2f}', 'average certainty when correct'),
        ('Sure and wrong', f'{conf[~correct].mean():.2f}', 'average certainty when wrong'),
        ('Answers given confidently', f'{hi.mean() * 100:.0f}%', 'certainty above 0.9'),
        ('…of which are wrong', f'{err_hi:.0f}%',
         'how far confidence can be trusted', 'bad' if err_hi > 12 else ''))

    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    pairs = sorted(((i, j, cmn[i, j]) for i in range(sess.n_cls) for j in range(sess.n_cls)
                    if i != j and cm[i, j] > 0), key=lambda t: -t[2])[:6]
    rows = [[DATASETS[dataset]['names'][i], DATASETS[dataset]['names'][j], f'{f * 100:.0f}%']
            for i, j, f in pairs]
    return head, plots.confidence_fig(conf, correct), gr.update(value=gal, visible=bool(gal)), \
        rows, plots.confusion_fig(cm, dataset, sess.seen, sess.base_cls)


# ================================================================= shell
FONT = ['system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial',
        'sans-serif']
FONT_MONO = ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace']
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.teal,
                       secondary_hue=gr.themes.colors.emerald,
                       neutral_hue=gr.themes.colors.slate,
                       font=FONT, font_mono=FONT_MONO)

CSS = """
:root, .dark {
  --ground:#f4f8f6; --surface:#ffffff; --line:#dce8e3;
  --ink:#12302a; --ink-soft:#55706a; --accent:#0f8c7e; --accent-deep:#0a6358;
  --warn:#a8620c; --bad:#a83232;
  --body-background-fill:var(--ground) !important;
  --background-fill-primary:var(--surface) !important;
  --background-fill-secondary:var(--ground) !important;
  --block-background-fill:var(--surface) !important;
  --panel-background-fill:var(--surface) !important;
  --input-background-fill:var(--surface) !important;
  --border-color-primary:var(--line) !important;
  --block-label-background-fill:var(--ground) !important;
  --block-label-text-color:var(--ink-soft) !important;
  --block-title-text-color:var(--ink) !important;
  --body-text-color:var(--ink) !important;
  --body-text-color-subdued:var(--ink-soft) !important;
  --table-odd-background-fill:var(--surface) !important;
  --table-even-background-fill:var(--ground) !important;
  --color-accent-soft:#e3f2ef !important;
  color-scheme: light;
}
.dark { color-scheme: light; }
body, gradio-app { background:var(--ground) !important; color:var(--ink) !important; }

/* centre the whole app - without this the container hugs the left edge on wide screens */
.gradio-container { max-width:1180px !important; margin:0 auto !important;
                    padding:0 24px 40px !important; }
.gradio-container p, .gradio-container li { font-size:14px; line-height:1.55; }

#hero { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
        border-bottom:2px solid var(--accent-deep); padding:6px 0 12px; }
#hero h1 { font-size:26px; font-weight:650; margin:0; color:var(--accent-deep);
           letter-spacing:-.02em; }
#hero span { color:var(--ink-soft); font-size:13.5px; }
.warnbar { background:#fdf8ec; border:1px solid #ecd9ad; border-left:3px solid #b3860d;
           border-radius:6px; padding:7px 12px; font-size:12px; color:#6a4d07;
           margin:10px 0 2px; }
.notice { background:var(--ground); border:1px dashed var(--line); border-radius:8px;
          padding:16px; color:var(--ink-soft); font-size:14px; }
.banner { background:#e9f4f1; border:1px solid #bfded6; border-radius:8px; padding:10px 14px;
          font-size:14px; color:var(--ink); margin:2px 0 6px; }
.banner.good { background:#e7f5ee; border-color:#b6ddc9; }

.card-row { display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 4px; }
.metric { flex:1 1 130px; padding:10px 13px; border:1px solid var(--line);
          border-radius:9px; background:var(--surface); }
.metric.good { border-color:#9fd3c4; background:#f2fbf8; }
.metric.bad  { border-color:#e6b9b9; background:#fdf4f4; }
.metric .k { font-size:10.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
             color:var(--ink-soft); }
.metric .v { font-size:24px; font-weight:650; color:var(--accent-deep); margin-top:4px;
             font-variant-numeric:tabular-nums; line-height:1.15; }
.metric.bad .v { color:var(--bad); }
.metric .s { font-size:11px; color:var(--ink-soft); margin-top:2px; }

.chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.chip { display:inline-block; padding:4px 9px; border-radius:20px; font-size:12px;
        font-weight:500; border:1px solid transparent; }
.chip.base  { background:#e4f2ee; color:#0a6358; border-color:#b9dcd3; }
.chip.novel { background:#fdeee1; color:#8a4a11; border-color:#f0cfae; }
.chip.todo  { background:transparent; color:var(--ink-soft);
              border-color:var(--line); border-style:dashed; }
.chip-key { font-size:11.5px; color:var(--ink-soft); margin-top:6px;
            display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.chip-key .chip { width:16px; height:12px; padding:0; border-radius:3px; margin-left:10px; }
.chip-key .chip:first-child { margin-left:0; }

.decision { border:1px solid var(--line); border-left:4px solid var(--accent);
            border-radius:9px; padding:14px 18px; background:var(--surface); }
.decision.warn { border-left-color:var(--warn); }
.decision.bad  { border-left-color:var(--bad); }
.decision .dk { font-size:12.5px; color:var(--ink-soft); }
.decision .db { font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
                color:var(--accent); margin-top:6px; }
.decision.warn .db { color:var(--warn); }
.decision.bad .db  { color:var(--bad); }
.decision .dv { font-size:20px; font-weight:640; color:var(--ink); margin-top:4px;
                line-height:1.3; }

.gradio-container .table-wrap, .gradio-container table,
.gradio-container table td, .gradio-container table th,
.gradio-container .cell-wrap span, .gradio-container .cell-wrap input {
  font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif !important;
  font-size:13px !important; font-variant-numeric:tabular-nums;
}
.gradio-container table th { font-weight:600; color:var(--ink-soft); }
"""


with gr.Blocks(title='MedFSCIL Studio') as demo:
    gr.HTML("<div id='hero'><h1>MedFSCIL Studio</h1>"
            "<span>Teach a medical image model a new disease from a handful of "
            "examples — without retraining it.</span></div>")
    gr.HTML(f"<div class='warnbar'>{DISCLAIMER}</div>")

    # ------------------------------------------------------------- TAB 1
    with gr.Tab('Teach a disease'):
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                d1 = gr.Dropdown(READY, value=READY[0], label='Image collection')
                b1 = gr.CheckboxGroup(backbone_choices(READY[0]),
                                      value=backbone_choices(READY[0])[:1], label='Model')
                f1 = gr.Dropdown(list(FUSION_MODES), value=list(FUSION_MODES)[0],
                                 label='How to combine them', interactive=False)
                k1 = gr.Slider(1, 20, value=K_SHOT, step=1, label='Examples per new disease')
                s1 = gr.Slider(0, 200, value=SEED0, step=1, label='Which examples get picked')
                c1 = gr.Dropdown(list(CALIB), value=list(CALIB)[0], label='Correction method')
                nxt = gr.Button('Teach the next disease', variant='primary')
                alla = gr.Button('Teach all remaining')
                reset = gr.Button('Start over')
            with gr.Column(scale=2, min_width=430):
                banner1 = gr.HTML()
                cards1 = gr.HTML()
                chips1 = gr.HTML()
                gal1 = gr.Gallery(label='The examples it just learned from', columns=5,
                                  height=170, object_fit='contain', visible=False)
        with gr.Row():
            curve = gr.Plot(show_label=False)
            pcbar = gr.Plot(show_label=False)
        cmplot = gr.Plot(show_label=False)
        with gr.Accordion('How this works', open=False):
            gr.Markdown(
                'The model is never retrained. It turns each image into a list of numbers, '
                'and represents a disease by the average of those numbers across its '
                'examples. Recognising a new image means finding the closest average. '
                'Adding a disease is therefore a single averaging step, which is why the '
                'timer reads in microseconds.\n\n'
                'It starts already familiar with the most common diseases in the collection, '
                'having seen thousands of examples of each. Every further disease arrives '
                'with only the handful you choose — that is the hard part, and the gap '
                'between the two accuracy figures is what this field is trying to close.\n\n'
                '**The charts.** The first tracks accuracy as diseases are added. The second '
                'shows how well each individual disease is recognised, most common on the '
                'left. The third is a grid of what gets mistaken for what: each row is a true '
                'disease and adds up to 100%, so the diagonal is how often it is recognised, '
                'and bright squares off the diagonal are systematic mix-ups.\n\n'
                '**Worth trying.** Move *which examples get picked* — on the rarest diseases '
                'the result swings by ten points depending on which few images you happen to '
                'get. And notice that some diseases score badly even though the model saw '
                'thousands of them, which means the difficulty is not only about scarcity.')

        sess_st, hist_st = gr.State(None), gr.State([])
        out1 = [sess_st, hist_st, cards1, banner1, chips1, curve, pcbar, cmplot, gal1]
        cfg1 = [d1, b1, f1, c1, k1, s1]
        learn_in = [sess_st, hist_st, k1, d1, b1, f1, c1, s1]

        nxt.click(learn_next, learn_in, out1)
        alla.click(learn_all, learn_in, out1)
        reset.click(build_session, cfg1, out1)
        for ctrl in (d1, b1, f1, c1):
            ctrl.change(build_session, cfg1, out1)
        for sl in (k1, s1):
            sl.release(build_session, cfg1, out1)

        d1.change(lambda d: (gr.update(choices=backbone_choices(d),
                                       value=backbone_choices(d)[:1]),
                             gr.update(interactive=False)), d1, [b1, f1])
        b1.change(lambda bb: gr.update(interactive=len(bb or []) > 1), b1, f1)
        demo.load(build_session, cfg1, out1)

    # ------------------------------------------------------------- TAB 2
    with gr.Tab('Which model to use'):
        d2 = gr.Dropdown(READY, value=READY[0], label='Image collection')
        dec2 = gr.HTML()
        tbl2 = gr.Dataframe(headers=['Model', 'Scores now', 'Its features could support',
                                     'Lost by the matching step'],
                            show_label=False, wrap=True, interactive=False)
        plot2 = gr.Plot(show_label=False)
        with gr.Accordion('Why', open=False):
            why2 = gr.Markdown()

        def do_route(dataset):
            r = route(dataset)
            return (decision_card(r), evidence_rows(r), why_markdown(r),
                    plots.backbone_compare(r['live'], r['reported_advanced'],
                                           r['fused'], dataset))
        d2.change(do_route, d2, [dec2, tbl2, why2, plot2])
        demo.load(do_route, d2, [dec2, tbl2, why2, plot2])

    # ------------------------------------------------------------- TAB 3
    with gr.Tab('Combining models'):
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                d3 = gr.Dropdown(READY, value=READY[0], label='Image collection')
                b3 = gr.CheckboxGroup(backbone_choices(READY[0]),
                                      value=backbone_choices(READY[0]), label='Models to combine')
                f3 = gr.Dropdown(list(FUSION_MODES), value=list(FUSION_MODES)[0],
                                 label='How to combine them')
                go3 = gr.Button('Measure', variant='primary')
            with gr.Column(scale=2, min_width=430):
                fus_cards = gr.HTML("<div class='notice'>Press <b>Measure</b> to test this "
                                    "combination on the held-out images.</div>")
                singles_tbl = gr.Dataframe(headers=['Model on its own', 'Accuracy'],
                                           show_label=False, wrap=True, interactive=False)
        dis_plot = gr.Plot(show_label=False)

        with gr.Row():
            dis_btn = gr.Button('Show images they read differently')
            dis_n = gr.Slider(6, 24, value=12, step=2, label='How many')
        dis_head = gr.HTML()
        dis_gal = gr.Gallery(show_label=False, columns=6, height=210, object_fit='contain',
                            visible=False)
        dis_tbl = gr.Dataframe(show_label=False, wrap=True, interactive=False)

        sweep_btn = gr.Button('Try every combination')
        sweep_tbl = gr.Dataframe(
            headers=['Combination', 'How combined', 'Accuracy %', 'Balanced %',
                     'Starting classes %', 'Added classes %'],
            show_label=False, wrap=True, interactive=False)

        with gr.Accordion('Why combining helps', open=False):
            gr.Markdown(
                'Different models fail on different images. The chart above counts how often '
                'two of them give different answers for the same image — the higher that is, '
                'the more they are seeing different things, and the more there is to gain by '
                'putting them together.\n\n'
                'The **ceiling** is the share of images that at least one model already gets '
                'right. No way of combining them can beat it, so the distance between the '
                'combined score and the ceiling is the headroom a smarter rule could still '
                'claim.\n\n'
                '*Join their features* merges what each model saw before deciding; the other '
                'two let each model decide first and then pool the votes.\n\n'
                'The orange marks on the chart are values recorded in earlier runs of this '
                'study, shown so the live measurement can be checked against them.')

        go3.click(fusion_report, [d3, b3, f3], [fus_cards, dis_plot, singles_tbl])
        dis_btn.click(disagreement_gallery, [d3, dis_n], [dis_head, dis_gal, dis_tbl])
        sweep_btn.click(fusion_sweep, d3, sweep_tbl)
        d3.change(lambda d: (gr.update(choices=backbone_choices(d), value=backbone_choices(d)),
                             gr.update(value=None), gr.update(visible=False),
                             gr.update(value=None), gr.update(value=None)),
                  d3, [b3, dis_head, dis_gal, dis_tbl, sweep_tbl])

    # ------------------------------------------------------------- TAB 4
    with gr.Tab('Where it goes wrong'):
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                d4 = gr.Dropdown(READY, value=READY[0], label='Image collection')
                b4 = gr.CheckboxGroup(backbone_choices(READY[0]),
                                      value=backbone_choices(READY[0])[:1], label='Model')
                f4 = gr.Dropdown(list(FUSION_MODES), value=list(FUSION_MODES)[0],
                                 label='How to combine them', interactive=False)
                n4 = gr.Slider(6, 40, value=12, step=2, label='How many mistakes to show')
                go4 = gr.Button('Examine mistakes', variant='primary')
            with gr.Column(scale=2, min_width=430):
                err_cards = gr.HTML("<div class='notice'>Press <b>Examine mistakes</b> to see "
                                    "what this model gets wrong, and how sure it was.</div>")
                pairs_tbl = gr.Dataframe(headers=['Actually is', 'Mistaken for', 'How often'],
                                         show_label=False, wrap=True, interactive=False)
        conf_plot = gr.Plot(show_label=False)
        err_gal = gr.Gallery(label='Its most confident mistakes — actual → guess · certainty',
                             columns=6, height=230, object_fit='contain', visible=False)
        cm4 = gr.Plot(show_label=False)
        with gr.Accordion('How to read this', open=False):
            gr.Markdown(
                'Every answer comes with a certainty score. Ideally the model is certain when '
                'it is right and hesitant when it is wrong, so that low certainty could be '
                'used to flag a case for a human.\n\n'
                'The **left chart** puts those two distributions side by side. Where they '
                'overlap, certainty cannot separate right answers from wrong ones.\n\n'
                'The **right chart** asks the practical question: if the model only answered '
                'the cases it was most sure about and passed the rest to a person, how '
                'accurate would those answers be? A steep rise to the left means the '
                'certainty score is worth acting on. A flat line means it is not.\n\n'
                'The gallery shows mistakes sorted by certainty, most confident first — these '
                'are the errors that would pass unchallenged.')

        go4.click(error_report, [d4, b4, f4, n4],
                  [err_cards, conf_plot, err_gal, pairs_tbl, cm4])
        d4.change(lambda d: (gr.update(choices=backbone_choices(d),
                                       value=backbone_choices(d)[:1]),
                             gr.update(interactive=False)), d4, [b4, f4])
        b4.change(lambda bb: gr.update(interactive=len(bb or []) > 1), b4, f4)

    # ------------------------------------------------------------- TAB 5
    with gr.Tab('Reference'):
        gr.Dataframe(
            value=[[DATASETS[d]['title'], DATASETS[d]['modality'], DATASETS[d]['source'],
                    f"{DATASETS[d]['n_base']} to start + "
                    f"{len(DATASETS[d]['names']) - DATASETS[d]['n_base']} added"]
                   for d in DATASETS],
            headers=['Collection', 'What the images are', 'Origin', 'Diseases'],
            label='Image collections', wrap=True, interactive=False)
        gr.Dataframe(
            value=[[BACKBONES[b]['title'], BACKBONES[b]['family'], f"{BACKBONES[b]['dim']}",
                    BACKBONES[b]['pretrain']] for b in BACKBONES],
            headers=['Model', 'Kind', 'Numbers per image', 'What it was trained on'],
            label='Models', wrap=True, interactive=False)

        _rows = []
        for d in DATASETS:
            for b, r in REPORTED_FROZEN.items():
                if r.get(d) is not None:
                    _rows.append([DATASETS[d]['title'], pretty(b), 'no retraining', r[d]])
            for b, r in REPORTED_ADVANCED.items():
                if r.get(d) is not None:
                    _rows.append([DATASETS[d]['title'], pretty(b), 'adapted (retrained)', r[d]])
            if d in REPORTED_FUSION:
                _rows.append([DATASETS[d]['title'], REPORTED_FUSION[d][1], 'combined',
                              REPORTED_FUSION[d][0]])
        gr.Dataframe(value=_rows, headers=['Collection', 'Model', 'Approach', 'Accuracy %'],
                     label='Stored results — averages of five runs, for comparison',
                     wrap=True, interactive=False)

        with gr.Accordion('How the numbers here are produced', open=False):
            gr.Markdown(
                'Within each collection the diseases are ordered by how many training images '
                'they have. The most common ones form the starting set and use all of their '
                'images; the rest are added one at a time with only the handful you choose.\n\n'
                'Each image is converted once into a list of numbers by a model whose weights '
                'never change. Those lists are adjusted onto a common scale, a disease is '
                'represented by the average of its examples, and an image is labelled by '
                'whichever average it sits closest to. A correction step nudges each new '
                'disease toward the ones it most resembles, which compensates for how noisy '
                'an average of five images is.\n\n'
                'Everything you press is measured on the spot, on held-out images the model '
                'was never given, using one draw of examples that you control. The **stored '
                'results** table averages five draws instead, so small differences from the '
                'live figures are expected. Its *adapted* rows come from a slower approach '
                'that does retrain part of the model; those are recorded values, not run '
                'here, and appear as striped bars on the model-comparison chart.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port, theme=THEME, css=CSS)
